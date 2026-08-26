"""HuggingFace model utilities for running the TDC-KV pipeline.

The core TDC-KV modules operate on token ids, observed attention, and KV-cache
tensors. This file provides the thin adapter layer from standard
`transformers` causal language models to those tensors.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import math
from typing import Any, Sequence

import torch

from src.core.chunker import FixedSizeChunkConstructor, SentenceBoundaryChunkConstructor
from src.core.dependency_graph import (
    SparseChunkDependencyGraph,
    SparseChunkDependencyGraphBuilder,
    aggregate_attention_rows,
)
from src.core.numerical import require_finite_tensor, validate_token_ids
from src.models.cache_manager import DecodingCacheManager
from src.models.hf_cache_adapter import build_dynamic_cache, cache_layer_tensors


@dataclass
class HfModelBundle:
    """Loaded HuggingFace model/tokenizer pair."""

    model: Any
    tokenizer: Any
    device: torch.device


@dataclass
class PreparedPrompt:
    """A model-ready prompt shared by FullKV and cache-prefill paths."""

    raw_text: str
    rendered_text: str
    serialization: str
    model_inputs: dict[str, torch.Tensor]
    original_token_count: int
    was_truncated: bool
    truncation_side: str

    @property
    def input_ids(self) -> torch.Tensor:
        return self.model_inputs["input_ids"]


@dataclass
class HfPrefillRecord:
    """Tensors extracted from one prompt prefill pass."""

    sample_id: str
    prompt: str
    input_ids: torch.Tensor
    chunks: list[torch.Tensor]
    chunk_map: torch.Tensor
    attention_obs: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    dependency_graph: SparseChunkDependencyGraph | None = None
    next_token_id: int | None = None
    next_token_logits: torch.Tensor | None = None
    max_chunk_tokens: int | None = None
    prefill_block_size: int | None = None
    prefill_blocks: int = 1

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.numel())


@dataclass(frozen=True)
class HfGenerationResult:
    """Generated FullKV continuation and its exact token IDs."""

    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class GreedyGenerationPolicy:
    """Deterministic logits/stopping contract shared by every decode path."""

    repetition_penalty: float
    eos_token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "greedy_model_repetition_penalty_v1",
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": float(self.repetition_penalty),
            "repetition_history": "full_prompt_plus_generated_tokens",
            "eos_token_ids": list(self.eos_token_ids),
            "stopping": "eos_or_max_new_tokens",
        }


@dataclass(frozen=True)
class EvictedGenerationResult:
    """Generated text plus bounded-cache diagnostics."""

    text: str
    cache_summary: dict[str, int]
    token_ids: tuple[int, ...]


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve `auto` to CUDA when available, otherwise CPU."""
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_torch_dtype(
    dtype: str = "auto",
    *,
    device: str | torch.device = "auto",
) -> torch.dtype | None:
    """Resolve a CLI dtype string for model loading."""
    dtype = dtype.lower()
    if dtype == "auto":
        # Preserve the checkpoint-native dtype. CUDA availability alone is not
        # evidence that FP16 is numerically safe for eager attention.
        return None
    if dtype in {"none", "default"}:
        return None
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype `{dtype}`.")


def resolve_greedy_generation_policy(
    model: Any,
    tokenizer: Any,
) -> GreedyGenerationPolicy:
    """Resolve the model's repetition penalty and EOS set for greedy decoding."""
    generation_config = getattr(model, "generation_config", None)
    penalty = getattr(generation_config, "repetition_penalty", 1.0)
    penalty = 1.0 if penalty is None else float(penalty)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("generation repetition_penalty must be finite and positive.")

    configured_eos = getattr(generation_config, "eos_token_id", None)
    if configured_eos is None:
        configured_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if configured_eos is None:
        configured_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(configured_eos, (list, tuple, set)):
        eos_token_ids = tuple(dict.fromkeys(int(value) for value in configured_eos))
    elif configured_eos is None:
        eos_token_ids = ()
    else:
        eos_token_ids = (int(configured_eos),)
    return GreedyGenerationPolicy(
        repetition_penalty=penalty,
        eos_token_ids=eos_token_ids,
    )


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor | Sequence[int],
    *,
    penalty: float,
) -> torch.Tensor:
    """Apply the Transformers repetition-penalty rule to complete token history."""
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("repetition penalty must be finite and positive.")
    require_finite_tensor("generation_logits_before_repetition_penalty", logits)
    squeeze_batch = logits.ndim == 1
    if squeeze_batch:
        scores = logits.unsqueeze(0).clone()
    elif logits.ndim == 2:
        scores = logits.clone()
    else:
        raise ValueError("generation logits must have shape [vocab] or [batch, vocab].")

    if isinstance(token_history, torch.Tensor):
        history = token_history.to(device=scores.device, dtype=torch.long)
    else:
        history = torch.tensor(
            list(token_history),
            dtype=torch.long,
            device=scores.device,
        )
    if history.ndim == 1:
        history = history.unsqueeze(0)
    if history.ndim != 2 or history.shape[0] != scores.shape[0]:
        raise ValueError("token history must have shape [batch, sequence].")
    if history.numel() == 0 or penalty == 1.0:
        return scores.squeeze(0) if squeeze_batch else scores
    validate_token_ids(
        history,
        vocab_size=int(scores.shape[-1]),
        name="generation_repetition_history",
    )

    gathered = torch.gather(scores, 1, history)
    adjusted = torch.where(
        gathered < 0,
        gathered * penalty,
        gathered / penalty,
    )
    scores.scatter_(1, history, adjusted)
    require_finite_tensor("generation_logits_after_repetition_penalty", scores)
    return scores.squeeze(0) if squeeze_batch else scores


def load_hf_model_and_tokenizer(
    model_name: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    device: str | torch.device = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    allow_attn_fallback: bool = False,
) -> HfModelBundle:
    """Load a HuggingFace causal LM and tokenizer lazily.

    `transformers` is imported inside this function so the rest of the repo can
    still run unit tests without the optional HF runtime installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "HuggingFace support requires `transformers`. Install dependencies "
            "from requirements.txt or environment.yml."
        ) from exc

    device_obj = resolve_device(device)
    torch_dtype = resolve_torch_dtype(dtype, device=device_obj)

    hub_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision:
        hub_kwargs["revision"] = revision
    if token:
        hub_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **hub_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = dict(hub_kwargs)
    if str(dtype).strip().lower() == "auto":
        model_kwargs["dtype"] = "auto"
    elif torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    load_attempts = [dict(model_kwargs)]
    if "dtype" in model_kwargs:
        legacy_dtype_kwargs = dict(model_kwargs)
        legacy_dtype_kwargs["torch_dtype"] = legacy_dtype_kwargs.pop("dtype")
        load_attempts.append(legacy_dtype_kwargs)
    if attn_implementation and allow_attn_fallback:
        for candidate in list(load_attempts):
            without_attn = dict(candidate)
            without_attn.pop("attn_implementation", None)
            load_attempts.append(without_attn)

    last_type_error: TypeError | None = None
    for kwargs in load_attempts:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            break
        except TypeError as exc:
            last_type_error = exc
    else:
        assert last_type_error is not None
        raise last_type_error

    model.to(device_obj)
    model.eval()
    if attn_implementation and not allow_attn_fallback:
        actual_attn = getattr(model.config, "_attn_implementation", None)
        if str(actual_attn) != str(attn_implementation):
            raise RuntimeError(
                "Requested attention implementation was not honored: "
                f"requested={attn_implementation!r}, actual={actual_attn!r}."
            )
    return HfModelBundle(model=model, tokenizer=tokenizer, device=device_obj)


def model_device(model: Any) -> torch.device:
    """Best-effort device lookup for a loaded torch module."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def prepare_prompt(
    *,
    tokenizer: Any,
    prompt: str,
    max_length: int | None = None,
    serialization: str = "auto",
    truncation_side: str = "right",
) -> PreparedPrompt:
    """Render and tokenize a prompt once for every generation/cache path."""
    requested = str(serialization).strip().lower()
    if requested not in {"auto", "raw", "chat"}:
        raise ValueError("Prompt serialization must be `auto`, `raw`, or `chat`.")

    has_chat_template = bool(getattr(tokenizer, "chat_template", None)) and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )
    resolved = "chat" if requested == "auto" and has_chat_template else requested
    if resolved == "auto":
        resolved = "raw"
    if resolved == "chat" and not has_chat_template:
        raise ValueError(
            "Chat prompt serialization was requested, but the tokenizer has no "
            "chat template. Use `raw` or a tokenizer with `apply_chat_template`."
        )
    truncation_side = str(truncation_side).strip().lower()
    if truncation_side not in {"left", "right"}:
        raise ValueError("truncation_side must be `left` or `right`.")
    if max_length is not None and int(max_length) <= 0:
        raise ValueError("max_length must be positive when provided.")

    raw_text = str(prompt)
    if resolved == "chat":
        rendered_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        add_special_tokens = False
    else:
        rendered_text = raw_text
        add_special_tokens = True

    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "add_special_tokens": add_special_tokens,
    }
    encoded = tokenizer(rendered_text, **tokenizer_kwargs)
    model_inputs = {
        key: value.detach().cpu().clone()
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    input_ids = model_inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Prepared prompts require tokenizer input_ids with batch size 1.")
    if input_ids.shape[1] == 0:
        raise ValueError("Prompt serialization produced no input tokens.")
    original_token_count = int(input_ids.shape[1])
    was_truncated = max_length is not None and original_token_count > int(max_length)
    if was_truncated:
        width = int(max_length)
        model_inputs = {
            key: (
                value[..., -width:]
                if truncation_side == "left"
                and value.ndim >= 2
                and value.shape[-1] == original_token_count
                else value[..., :width]
                if value.ndim >= 2 and value.shape[-1] == original_token_count
                else value
            )
            for key, value in model_inputs.items()
        }

    return PreparedPrompt(
        raw_text=raw_text,
        rendered_text=str(rendered_text),
        serialization=resolved,
        model_inputs=model_inputs,
        original_token_count=original_token_count,
        was_truncated=bool(was_truncated),
        truncation_side=truncation_side,
    )


def _prepared_inputs_on_device(
    prepared_prompt: PreparedPrompt,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in prepared_prompt.model_inputs.items()
    }


def build_position_kwargs(
    model: Any,
    position_ids: torch.Tensor,
    *,
    include_cache_position: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Build logical-position arguments for supported Transformers versions.

    Transformers 4 model families may expose both ``position_ids`` and
    ``cache_position``. Transformers 5 Llama/Qwen forwards use
    ``position_ids`` and no longer expose ``cache_position``. Always preserve
    the global logical position through ``position_ids`` and provide the
    legacy cache argument only when it is an explicit forward parameter.
    """
    if position_ids.ndim != 2:
        raise ValueError("position_ids must have shape [batch, sequence].")

    if include_cache_position is None:
        include_cache_position = (
            "cache_position" in inspect.signature(model.forward).parameters
        )

    kwargs = {"position_ids": position_ids}
    if include_cache_position:
        kwargs["cache_position"] = position_ids.reshape(-1)
    return kwargs


def _normalize_layer_index(layer_index: int, num_layers: int) -> int:
    idx = int(layer_index)
    if idx < 0:
        idx = num_layers + idx
    if idx < 0 or idx >= num_layers:
        raise IndexError(
            f"layer_index {layer_index} is out of range for {num_layers} layers."
        )
    return idx


def _select_batch_zero(cache_tensor: torch.Tensor) -> torch.Tensor:
    if cache_tensor.ndim == 4:
        return cache_tensor[0]
    if cache_tensor.ndim == 3:
        return cache_tensor
    raise ValueError(
        f"Expected cache tensor shape [batch,heads,t,dim] or [heads,t,dim], "
        f"got {tuple(cache_tensor.shape)}."
    )


def extract_layer_kv_cache(
    past_key_values: Any,
    *,
    layer_index: int = -1,
    offload_to_cpu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one layer's KV cache as `[heads, seq_len, head_dim]` tensors."""
    cache_layers = cache_layer_tensors(past_key_values)
    layer_count = len(cache_layers)
    idx = _normalize_layer_index(layer_index, layer_count)
    key_tensor, value_tensor = cache_layers[idx]

    k_cache = _select_batch_zero(key_tensor.detach())
    v_cache = _select_batch_zero(value_tensor.detach())
    if offload_to_cpu:
        k_cache = k_cache.cpu()
        v_cache = v_cache.cpu()
    return k_cache, v_cache


def extract_full_kv_cache(
    past_key_values: Any,
    *,
    offload_to_cpu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract all layers' KV cache as `[num_layers, batch, heads, seq_len, head_dim]` tensors."""
    cache_layers = cache_layer_tensors(past_key_values)
    k_layers = []
    v_layers = []
    
    for key_tensor, value_tensor in cache_layers:
        k_cache = key_tensor.detach()
        v_cache = value_tensor.detach()
        
        if offload_to_cpu:
            k_cache = k_cache.cpu()
            v_cache = v_cache.cpu()
            
        k_layers.append(k_cache)
        v_layers.append(v_cache)
        
    return torch.stack(k_layers, dim=0), torch.stack(v_layers, dim=0)


def _select_attention_window(
    attention: torch.Tensor,
    *,
    window_size: int,
    offload_to_cpu: bool,
) -> torch.Tensor:
    if attention.ndim != 4:
        raise ValueError(
            f"Expected attention shape [batch,heads,queries,keys], got {tuple(attention.shape)}."
        )
    _, _, query_len, key_len = attention.shape
    obs_len = min(max(1, int(window_size)), int(query_len), int(key_len))
    selected = attention[0, :, -obs_len:, :key_len].detach().to(torch.float32)
    return selected.cpu() if offload_to_cpu else selected


def extract_attention_obs(
    attentions: Sequence[torch.Tensor],
    *,
    window_size: int,
    mode: str = "last",
    layer_index: int = -1,
    offload_to_cpu: bool = True,
) -> torch.Tensor:
    """Extract observed attention as `[H,w,t]` or `[L,H,w,t]`."""
    if not attentions:
        raise ValueError(
            "Model output did not include attentions. Use output_attentions=True "
            "and, for some models, --attn-implementation eager."
        )

    mode = mode.lower()
    if mode == "last":
        idx = _normalize_layer_index(layer_index, len(attentions))
        return _select_attention_window(
            attentions[idx], window_size=window_size, offload_to_cpu=offload_to_cpu
        )
    if mode == "all":
        layers = [
            _select_attention_window(
                attention, window_size=window_size, offload_to_cpu=offload_to_cpu
            )
            for attention in attentions
        ]
        return torch.stack(layers, dim=0)
    raise ValueError("attention mode must be `last` or `all`.")


def _select_block_attention_rows(
    attentions: Sequence[torch.Tensor],
    *,
    query_start: int,
    query_end: int,
    key_start: int,
    sequence_length: int,
    mode: str,
    layer_index: int,
    offload_to_cpu: bool,
) -> torch.Tensor:
    """Select observation rows from one prefill block and pad logical key positions."""
    if not attentions:
        raise ValueError(
            "Model output did not include attentions. Use output_attentions=True "
            "and, for some models, --attn-implementation eager."
        )

    mode = mode.lower()
    if mode == "last":
        selected_layers = [attentions[_normalize_layer_index(layer_index, len(attentions))]]
    elif mode == "all":
        selected_layers = list(attentions)
    else:
        raise ValueError("attention mode must be `last` or `all`.")

    collected_layers: list[torch.Tensor] = []
    expected_shape: tuple[int, int, int] | None = None
    for attention in selected_layers:
        if attention is None:
            raise ValueError(
                "A selected attention layer is missing. Use an attention "
                "implementation that supports output_attentions=True."
            )
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(
                "Expected attention shape [1,heads,queries,keys], got "
                f"{tuple(attention.shape)}."
            )
        rows = attention[0, :, query_start:query_end, :].detach().to(torch.float32)
        key_end = key_start + int(rows.shape[-1])
        if key_start < 0 or key_end > sequence_length:
            raise ValueError("Attention key positions exceed the tokenized prompt length.")
        rows = torch.nn.functional.pad(
            rows,
            (key_start, sequence_length - key_end),
        )
        if expected_shape is None:
            expected_shape = tuple(rows.shape)
        elif tuple(rows.shape) != expected_shape:
            raise ValueError("All selected attention layers must have matching shapes.")
        collected_layers.append(rows.cpu() if offload_to_cpu else rows)

    if mode == "last":
        return collected_layers[0]
    return torch.stack(collected_layers, dim=0)


def run_hf_prefill(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    sample_id: str,
    observation_window: int,
    max_length: int | None = None,
    attention_mode: str = "last",
    layer_index: int = -1,
    layer_weighting: str = "linear",
    min_chunk_tokens: int = 5,
    max_chunk_tokens: int = 64,
    dependency_top_k: int | None = 8,
    prefill_block_size: int = 128,
    chunk_constructor: (
        SentenceBoundaryChunkConstructor | FixedSizeChunkConstructor | None
    ) = None,
    offload_to_cpu: bool = True,
    prepared_prompt: PreparedPrompt | None = None,
) -> HfPrefillRecord:
    """Run bounded-attention blockwise prefill and return TDC-KV-ready tensors."""
    if prefill_block_size <= 0:
        raise ValueError("prefill_block_size must be positive.")
    if observation_window <= 0:
        raise ValueError("observation_window must be positive.")

    device = model_device(model)
    if prepared_prompt is None:
        prepared_prompt = prepare_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            serialization="raw",
        )
    elif prepared_prompt.raw_text != str(prompt):
        raise ValueError("Prepared prompt does not match the supplied raw prompt.")
    encoded = _prepared_inputs_on_device(prepared_prompt, device)

    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported by the experiment runner.")
    if input_ids.shape[1] == 0:
        raise ValueError(f"Sample `{sample_id}` produced no input tokens.")

    input_ids_cpu = input_ids[0].detach().cpu()
    if chunk_constructor is None:
        chunk_constructor = SentenceBoundaryChunkConstructor(
            tokenizer=tokenizer,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            device="cpu",
        )
    chunks, chunk_map = chunk_constructor.forward(input_ids_cpu)
    if chunk_map.ndim != 1 or chunk_map.numel() != input_ids_cpu.numel():
        raise ValueError("Chunk constructor must return one chunk id per input token.")

    sequence_length = int(input_ids.shape[1])
    block_size = min(int(prefill_block_size), sequence_length)
    observation_start = max(0, sequence_length - int(observation_window))
    graph_device = torch.device("cpu") if offload_to_cpu else device
    graph_builder = None
    if dependency_top_k is not None:
        graph_builder = SparseChunkDependencyGraphBuilder(
            num_chunks=len(chunks),
            top_k=int(dependency_top_k),
            device=graph_device,
        )

    observation_blocks: list[torch.Tensor] = []
    past_key_values = None
    next_token_id = None
    next_token_logits = None
    include_cache_position = (
        "cache_position" in inspect.signature(model.forward).parameters
    )
    prefill_blocks = 0
    generation_policy = resolve_greedy_generation_policy(model, tokenizer)

    with torch.no_grad():
        for start in range(0, sequence_length, block_size):
            end = min(start + block_size, sequence_length)
            position_ids = torch.arange(start, end, dtype=torch.long, device=device)[None, :]
            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids[:, start:end],
                "use_cache": True,
                "output_attentions": True,
                "return_dict": True,
            }
            forward_kwargs.update(
                build_position_kwargs(
                    model,
                    position_ids,
                    include_cache_position=include_cache_position,
                )
            )
            if "attention_mask" in encoded:
                forward_kwargs["attention_mask"] = encoded["attention_mask"][:, :end]
            if past_key_values is not None:
                forward_kwargs["past_key_values"] = past_key_values
            for key, value in encoded.items():
                if key in {"input_ids", "attention_mask", "position_ids"}:
                    continue
                if value.ndim >= 2 and value.shape[0] == 1 and value.shape[-1] == sequence_length:
                    forward_kwargs[key] = value[..., start:end]
                else:
                    forward_kwargs[key] = value

            outputs = model(**forward_kwargs)
            attentions = outputs.attentions
            if outputs.logits is None:
                raise ValueError("Model output did not include prefill logits.")
            require_finite_tensor(
                "prefill_logits",
                outputs.logits,
                stage="prefill",
                sample_id=sample_id,
                block_index=prefill_blocks,
                block_start=start,
                block_end=end,
            )
            if not attentions:
                raise ValueError(
                    "Model output did not include attentions. Use an attention "
                    "implementation that supports output_attentions=True."
                )

            first_attention = next(
                (attention for attention in attentions if attention is not None),
                None,
            )
            if first_attention is None:
                raise ValueError("Every model attention layer returned None.")
            for attention_layer, attention in enumerate(attentions):
                if attention is not None:
                    require_finite_tensor(
                        "prefill_attention",
                        attention,
                        stage="prefill",
                        sample_id=sample_id,
                        block_index=prefill_blocks,
                        layer=attention_layer,
                    )
            query_len = int(first_attention.shape[-2])
            key_len = int(first_attention.shape[-1])
            if query_len != end - start:
                raise ValueError(
                    "Model attention query length does not match the current prefill block."
                )
            key_start = end - key_len
            if key_start < 0:
                raise ValueError("Model attention key length exceeds the processed prefix.")

            if graph_builder is not None:
                graph_rows = aggregate_attention_rows(
                    attentions,
                    mode=attention_mode,
                    layer_index=layer_index,
                    layer_weighting=layer_weighting,
                )
                graph_builder.update(
                    graph_rows,
                    query_chunk_ids=chunk_map[start:end],
                    key_chunk_ids=chunk_map[key_start:end],
                )

            overlap_start = max(start, observation_start)
            if overlap_start < end:
                observation_blocks.append(
                    _select_block_attention_rows(
                        attentions,
                        query_start=overlap_start - start,
                        query_end=end - start,
                        key_start=key_start,
                        sequence_length=sequence_length,
                        mode=attention_mode,
                        layer_index=layer_index,
                        offload_to_cpu=offload_to_cpu,
                    )
                )

            past_key_values = outputs.past_key_values
            for cache_layer, (key, value) in enumerate(
                cache_layer_tensors(past_key_values)
            ):
                require_finite_tensor(
                    "prefill_key_slice",
                    key[..., -query_len:, :],
                    stage="prefill",
                    sample_id=sample_id,
                    block_index=prefill_blocks,
                    layer=cache_layer,
                )
                require_finite_tensor(
                    "prefill_value_slice",
                    value[..., -query_len:, :],
                    stage="prefill",
                    sample_id=sample_id,
                    block_index=prefill_blocks,
                    layer=cache_layer,
                )
            final_logits = outputs.logits[0, -1, :]
            require_finite_tensor(
                "prefill_next_token_logits",
                final_logits,
                stage="prefill_argmax",
                sample_id=sample_id,
                block_index=prefill_blocks,
            )
            processed_final_logits = apply_repetition_penalty(
                final_logits,
                input_ids[:, :end],
                penalty=generation_policy.repetition_penalty,
            )
            next_token_id = int(torch.argmax(processed_final_logits).item())
            next_token_logits = processed_final_logits.detach().cpu().clone()
            validate_token_ids(
                [next_token_id],
                vocab_size=getattr(getattr(model, "config", None), "vocab_size", None),
                name="prefill_next_token_id",
                sample_id=sample_id,
            )
            prefill_blocks += 1
            if graph_builder is not None:
                del graph_rows
            del attentions, outputs

    if not observation_blocks:
        raise RuntimeError("No observation attention rows were collected during prefill.")
    attention_query_axis = 2 if attention_mode.lower() == "all" else 1
    attention_obs = torch.cat(observation_blocks, dim=attention_query_axis)
    require_finite_tensor(
        "attention_observation",
        attention_obs,
        stage="prefill_finalize",
        sample_id=sample_id,
    )
    expected_observations = min(int(observation_window), sequence_length)
    if int(attention_obs.shape[attention_query_axis]) != expected_observations:
        raise RuntimeError("Blockwise prefill collected an incomplete observation window.")

    dependency_graph = graph_builder.finalize() if graph_builder is not None else None
    if dependency_graph is not None and offload_to_cpu:
        dependency_graph = dependency_graph.to("cpu")
    k_cache, v_cache = extract_full_kv_cache(
        past_key_values,
        offload_to_cpu=offload_to_cpu,
    )
    require_finite_tensor(
        "prefill_k_cache", k_cache, stage="prefill_finalize", sample_id=sample_id
    )
    require_finite_tensor(
        "prefill_v_cache", v_cache, stage="prefill_finalize", sample_id=sample_id
    )
    if int(k_cache.shape[-2]) != sequence_length:
        raise ValueError(
            "The model returned a sliding or truncated KV cache. TDC-KV currently "
            "requires a full prompt cache before eviction."
        )

    return HfPrefillRecord(
        sample_id=sample_id,
        prompt=prepared_prompt.rendered_text,
        input_ids=input_ids_cpu,
        chunks=chunks,
        chunk_map=chunk_map,
        attention_obs=attention_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        dependency_graph=dependency_graph,
        next_token_id=next_token_id,
        next_token_logits=next_token_logits,
        max_chunk_tokens=chunk_constructor.max_chunk_tokens,
        prefill_block_size=block_size,
        prefill_blocks=prefill_blocks,
    )


def generate_text(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    max_length: int | None = None,
    return_details: bool = False,
    prepared_prompt: PreparedPrompt | None = None,
) -> str | HfGenerationResult:
    """Generate a deterministic continuation from the base HF model."""
    if max_new_tokens <= 0:
        empty = HfGenerationResult(text="", token_ids=())
        return empty if return_details else empty.text

    device = model_device(model)
    if prepared_prompt is None:
        prepared_prompt = prepare_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            serialization="raw",
        )
    elif prepared_prompt.raw_text != str(prompt):
        raise ValueError("Prepared prompt does not match the supplied raw prompt.")
    encoded = _prepared_inputs_on_device(prepared_prompt, device)

    try:
        from transformers import LogitsProcessor, LogitsProcessorList
    except ModuleNotFoundError as exc:  # pragma: no cover - model already requires HF
        raise ModuleNotFoundError(
            "HuggingFace generation requires `transformers`."
        ) from exc

    class _FiniteLogitsProcessor(LogitsProcessor):
        def __init__(self) -> None:
            self.step = 0

        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
        ) -> torch.FloatTensor:
            require_finite_tensor(
                "native_generation_logits",
                scores,
                stage="native_generation_argmax",
                decode_step=self.step,
            )
            self.step += 1
            return scores

    generation_policy = resolve_greedy_generation_policy(model, tokenizer)
    eos_token_id: int | list[int] | None
    if len(generation_policy.eos_token_ids) == 1:
        eos_token_id = generation_policy.eos_token_ids[0]
    elif generation_policy.eos_token_ids:
        eos_token_id = list(generation_policy.eos_token_ids)
    else:
        eos_token_id = None

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            repetition_penalty=generation_policy.repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([_FiniteLogitsProcessor()]),
        )

    prompt_len = int(encoded["input_ids"].shape[1])
    continuation = generated[0, prompt_len:]
    token_ids = validate_token_ids(
        continuation,
        vocab_size=getattr(getattr(model, "config", None), "vocab_size", None),
        name="native_generated_token_ids",
    )
    result = HfGenerationResult(
        text=tokenizer.decode(continuation, skip_special_tokens=True).strip(),
        token_ids=token_ids,
    )
    return result if return_details else result.text


@contextmanager
def extended_rotary_position_capacity(
    model: Any,
    *,
    required_sequence_length: int,
):
    """Temporarily extend legacy RoPE tables for logically positioned compact caches."""
    if required_sequence_length <= 0:
        raise ValueError("required_sequence_length must be positive.")

    original_forwards = []
    for module in model.modules():
        if "RotaryEmbedding" not in module.__class__.__name__:
            continue
        original_forward = module.forward
        if "seq_len" not in inspect.signature(original_forward).parameters:
            continue

        def make_patched_forward(bound_forward):
            def patched_forward(self, x, *args, seq_len=None, **kwargs):
                if seq_len is not None:
                    seq_len = max(int(seq_len), int(required_sequence_length))
                    return bound_forward(x, *args, seq_len=seq_len, **kwargs)
                return bound_forward(x, *args, **kwargs)

            return patched_forward

        original_forwards.append((module, original_forward))
        module.forward = make_patched_forward(original_forward).__get__(
            module,
            module.__class__,
        )

    try:
        yield
    finally:
        for module, original_forward in original_forwards:
            module.forward = original_forward


def generate_text_with_evicted_cache(
    *,
    model: Any,
    tokenizer: Any,
    first_new_token_id: int,
    max_new_tokens: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    original_sequence_length: int,
    prompt_token_ids: torch.Tensor | Sequence[int] | None = None,
    budget: int | None = None,
    kept_indices: torch.Tensor | None = None,
    chunks: Sequence[torch.Tensor] | None = None,
    chunk_scores: torch.Tensor | None = None,
    mask_tiers: torch.Tensor | None = None,
    recent_window: int = 16,
    return_details: bool = False,
) -> str | EvictedGenerationResult:
    """Generate text dynamically using an already evicted KV cache."""
    if max_new_tokens <= 0:
        empty_result = EvictedGenerationResult(text="", cache_summary={}, token_ids=())
        return empty_result if return_details else empty_result.text

    device = model_device(model)
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    generation_policy = resolve_greedy_generation_policy(model, tokenizer)
    if prompt_token_ids is None:
        if generation_policy.repetition_penalty != 1.0:
            raise ValueError(
                "prompt_token_ids is required when repetition_penalty is not 1.0."
            )
        prompt_history: tuple[int, ...] = ()
    else:
        prompt_history = validate_token_ids(
            prompt_token_ids,
            vocab_size=vocab_size,
            name="decode_prompt_token_ids",
            stage="decode_init",
        )
    validate_token_ids(
        [first_new_token_id],
        vocab_size=vocab_size,
        name="first_new_token_id",
        stage="decode_init",
    )
    require_finite_tensor("decode_initial_k_cache", k_cache, stage="decode_init")
    require_finite_tensor("decode_initial_v_cache", v_cache, stage="decode_init")
    cache_manager = None
    if budget is not None:
        if kept_indices is None:
            raise ValueError("kept_indices is required when a decode budget is set.")
        cache_manager = DecodingCacheManager.from_prompt(
            budget=int(budget),
            recent_window=int(recent_window),
            kept_indices=kept_indices,
            original_sequence_length=int(original_sequence_length),
            chunks=chunks,
            chunk_scores=chunk_scores,
            mask_tiers=mask_tiers,
        )
        k_cache, v_cache, _ = cache_manager.trim_cache_tensors(
            k_cache,
            v_cache,
            current_logical_length=int(original_sequence_length),
        )
    with extended_rotary_position_capacity(
        model,
        required_sequence_length=original_sequence_length + max_new_tokens,
    ):
        past_key_values = build_dynamic_cache(
            [
                (k_cache[layer_index].to(device), v_cache[layer_index].to(device))
                for layer_index in range(k_cache.shape[0])
            ]
        )

        input_ids = torch.tensor([[first_new_token_id]], dtype=torch.long, device=device)
        cache_len = k_cache.shape[-2]
        attention_mask = torch.ones(1, cache_len + 1, dtype=torch.long, device=device)
        position_ids = torch.tensor([[original_sequence_length]], dtype=torch.long, device=device)

        generated_tokens = [first_new_token_id]
        repetition_history = torch.tensor(
            [list(prompt_history) + generated_tokens],
            dtype=torch.long,
            device=device,
        )
        eos_token_ids = set(generation_policy.eos_token_ids)
        include_cache_position = (
            "cache_position" in inspect.signature(model.forward).parameters
        )

        if first_new_token_id in eos_token_ids:
            text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            cache_summary = cache_manager.summary() if cache_manager is not None else {}
            cache_summary["generated_tokens"] = len(generated_tokens)
            result = EvictedGenerationResult(
                text=text,
                cache_summary=cache_summary,
                token_ids=tuple(generated_tokens),
            )
            return result if return_details else result.text

        # We already generated the first token from the prefill step, so we need max_new_tokens - 1 more
        for decode_step in range(max_new_tokens - 1):
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            forward_kwargs.update(
                build_position_kwargs(
                    model,
                    position_ids,
                    include_cache_position=include_cache_position,
                )
            )
            with torch.no_grad():
                outputs = model(**forward_kwargs)

            if outputs.logits is None:
                raise ValueError("Model output did not include decode logits.")
            require_finite_tensor(
                "decode_logits",
                outputs.logits,
                stage="decode",
                decode_step=decode_step,
            )
            for cache_layer, (key, value) in enumerate(
                cache_layer_tensors(outputs.past_key_values)
            ):
                require_finite_tensor(
                    "decode_key_slice",
                    key[..., -1:, :],
                    stage="decode",
                    decode_step=decode_step,
                    layer=cache_layer,
                )
                require_finite_tensor(
                    "decode_value_slice",
                    value[..., -1:, :],
                    stage="decode",
                    decode_step=decode_step,
                    layer=cache_layer,
                )

            if cache_manager is not None:
                processed_position = int(position_ids.item())
                cache_manager.append_generated_token(
                    logical_position=processed_position
                )
                past_key_values, _ = cache_manager.trim_past_key_values(
                    outputs.past_key_values,
                    current_logical_length=processed_position + 1,
                )
            else:
                past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            require_finite_tensor(
                "decode_next_token_logits",
                next_token_logits,
                stage="decode_argmax",
                decode_step=decode_step,
            )
            processed_next_token_logits = apply_repetition_penalty(
                next_token_logits,
                repetition_history,
                penalty=generation_policy.repetition_penalty,
            )
            next_token = torch.argmax(
                processed_next_token_logits,
                dim=-1,
                keepdim=True,
            )
            next_token_id = validate_token_ids(
                next_token,
                vocab_size=vocab_size,
                name="decode_next_token_id",
                decode_step=decode_step,
            )[0]
            generated_tokens.append(next_token_id)
            repetition_history = torch.cat(
                (repetition_history, next_token.to(dtype=torch.long)),
                dim=1,
            )
            
            # Match Transformers.generate(), which uses the model generation
            # configuration and may define more than one EOS token.
            if next_token_id in eos_token_ids:
                break
                
            input_ids = next_token
            if cache_manager is not None:
                attention_mask = torch.ones(
                    1,
                    cache_manager.cache_length + 1,
                    dtype=torch.long,
                    device=device,
                )
            else:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(1, 1, dtype=torch.long, device=device),
                    ],
                    dim=1,
                )
            position_ids = position_ids + 1

        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        cache_summary = cache_manager.summary() if cache_manager is not None else {}
        cache_summary["generated_tokens"] = len(generated_tokens)
        validated_tokens = validate_token_ids(
            generated_tokens,
            vocab_size=vocab_size,
            name="generated_token_ids",
            stage="decode_finalize",
        )
        result = EvictedGenerationResult(
            text=text,
            cache_summary=cache_summary,
            token_ids=validated_tokens,
        )
        return result if return_details else result.text


__all__ = [
    "GreedyGenerationPolicy",
    "HfModelBundle",
    "HfPrefillRecord",
    "HfGenerationResult",
    "EvictedGenerationResult",
    "apply_repetition_penalty",
    "build_position_kwargs",
    "extended_rotary_position_capacity",
    "extract_attention_obs",
    "extract_layer_kv_cache",
    "generate_text",
    "generate_text_with_evicted_cache",
    "load_hf_model_and_tokenizer",
    "model_device",
    "prepare_prompt",
    "resolve_greedy_generation_policy",
    "resolve_device",
    "resolve_torch_dtype",
    "run_hf_prefill",
]
