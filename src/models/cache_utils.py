"""HuggingFace model utilities for running the TDC-KV pipeline.

The core TDC-KV modules operate on token ids, observed attention, and KV-cache
tensors. This file provides the thin adapter layer from standard
`transformers` causal language models to those tensors.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from typing import Any, Sequence

import torch

from src.core.chunker import SentenceBoundaryChunkConstructor
from src.core.dependency_graph import (
    SparseChunkDependencyGraph,
    SparseChunkDependencyGraphBuilder,
    aggregate_attention_rows,
)
from src.models.cache_manager import DecodingCacheManager


@dataclass
class HfModelBundle:
    """Loaded HuggingFace model/tokenizer pair."""

    model: Any
    tokenizer: Any
    device: torch.device


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
    prefill_block_size: int | None = None
    prefill_blocks: int = 1

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.numel())


@dataclass(frozen=True)
class EvictedGenerationResult:
    """Generated text plus bounded-cache diagnostics."""

    text: str
    cache_summary: dict[str, int]


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
    device_obj = resolve_device(device)

    if dtype == "auto":
        return torch.float16 if device_obj.type == "cuda" else torch.float32
    if dtype in {"none", "default"}:
        return None
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype `{dtype}`.")


def load_hf_model_and_tokenizer(
    model_name: str,
    *,
    device: str | torch.device = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    load_attempts = [dict(model_kwargs)]
    if "dtype" in model_kwargs:
        legacy_dtype_kwargs = dict(model_kwargs)
        legacy_dtype_kwargs["torch_dtype"] = legacy_dtype_kwargs.pop("dtype")
        load_attempts.append(legacy_dtype_kwargs)
    if attn_implementation:
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
    return HfModelBundle(model=model, tokenizer=tokenizer, device=device_obj)


def model_device(model: Any) -> torch.device:
    """Best-effort device lookup for a loaded torch module."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _legacy_past_key_values(past_key_values: Any) -> Sequence[Any]:
    if past_key_values is None:
        raise ValueError("Model output did not include `past_key_values`.")
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return past_key_values


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
    legacy_cache = _legacy_past_key_values(past_key_values)
    layer_count = len(legacy_cache)
    idx = _normalize_layer_index(layer_index, layer_count)
    layer_cache = legacy_cache[idx]

    if isinstance(layer_cache, dict):
        key_tensor = layer_cache.get("key_states")
        if key_tensor is None:
            key_tensor = layer_cache.get("key")
        value_tensor = layer_cache.get("value_states")
        if value_tensor is None:
            value_tensor = layer_cache.get("value")
    else:
        key_tensor, value_tensor = layer_cache[:2]

    if key_tensor is None or value_tensor is None:
        raise ValueError("Unable to extract key/value tensors from model cache.")

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
    legacy_cache = _legacy_past_key_values(past_key_values)
    k_layers = []
    v_layers = []
    
    for layer_cache in legacy_cache:
        if isinstance(layer_cache, dict):
            key_tensor = layer_cache.get("key_states")
            if key_tensor is None:
                key_tensor = layer_cache.get("key")
            value_tensor = layer_cache.get("value_states")
            if value_tensor is None:
                value_tensor = layer_cache.get("value")
        else:
            key_tensor, value_tensor = layer_cache[:2]

        if key_tensor is None or value_tensor is None:
            raise ValueError("Unable to extract key/value tensors from model cache.")
            
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
    min_chunk_tokens: int = 5,
    dependency_top_k: int | None = 8,
    prefill_block_size: int = 128,
    chunk_constructor: SentenceBoundaryChunkConstructor | None = None,
    offload_to_cpu: bool = True,
) -> HfPrefillRecord:
    """Run bounded-attention blockwise prefill and return TDC-KV-ready tensors."""
    if prefill_block_size <= 0:
        raise ValueError("prefill_block_size must be positive.")
    if observation_window <= 0:
        raise ValueError("observation_window must be positive.")

    device = model_device(model)
    tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        tokenizer_kwargs.update({"truncation": True, "max_length": int(max_length)})
    encoded = tokenizer(prompt, **tokenizer_kwargs)
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }

    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported by the experiment runner.")
    if input_ids.shape[1] == 0:
        raise ValueError(f"Sample `{sample_id}` produced no input tokens.")

    input_ids_cpu = input_ids[0].detach().cpu()
    if chunk_constructor is None:
        chunk_constructor = SentenceBoundaryChunkConstructor(
            tokenizer=tokenizer, min_chunk_tokens=min_chunk_tokens, device="cpu"
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
    forward_parameters = inspect.signature(model.forward).parameters
    prefill_blocks = 0

    with torch.no_grad():
        for start in range(0, sequence_length, block_size):
            end = min(start + block_size, sequence_length)
            position_ids = torch.arange(start, end, dtype=torch.long, device=device)[None, :]
            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids[:, start:end],
                "position_ids": position_ids,
                "use_cache": True,
                "output_attentions": True,
                "return_dict": True,
            }
            if "attention_mask" in encoded:
                forward_kwargs["attention_mask"] = encoded["attention_mask"][:, :end]
            if past_key_values is not None:
                forward_kwargs["past_key_values"] = past_key_values
            if "cache_position" in forward_parameters:
                forward_kwargs["cache_position"] = position_ids.reshape(-1)

            for key, value in encoded.items():
                if key in {"input_ids", "attention_mask", "position_ids"}:
                    continue
                if value.ndim >= 2 and value.shape[0] == 1 and value.shape[-1] == sequence_length:
                    forward_kwargs[key] = value[..., start:end]
                else:
                    forward_kwargs[key] = value

            outputs = model(**forward_kwargs)
            attentions = outputs.attentions
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
            if outputs.logits is not None:
                next_token_id = int(torch.argmax(outputs.logits[0, -1, :]).item())
            prefill_blocks += 1
            if graph_builder is not None:
                del graph_rows
            del attentions, outputs

    if not observation_blocks:
        raise RuntimeError("No observation attention rows were collected during prefill.")
    attention_query_axis = 2 if attention_mode.lower() == "all" else 1
    attention_obs = torch.cat(observation_blocks, dim=attention_query_axis)
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
    if int(k_cache.shape[-2]) != sequence_length:
        raise ValueError(
            "The model returned a sliding or truncated KV cache. TDC-KV currently "
            "requires a full prompt cache before eviction."
        )

    return HfPrefillRecord(
        sample_id=sample_id,
        prompt=prompt,
        input_ids=input_ids_cpu,
        chunks=chunks,
        chunk_map=chunk_map,
        attention_obs=attention_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        dependency_graph=dependency_graph,
        next_token_id=next_token_id,
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
) -> str:
    """Generate a deterministic continuation from the base HF model."""
    if max_new_tokens <= 0:
        return ""

    device = model_device(model)
    tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        tokenizer_kwargs.update({"truncation": True, "max_length": int(max_length)})
    encoded = tokenizer(prompt, **tokenizer_kwargs)
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = int(encoded["input_ids"].shape[1])
    continuation = generated[0, prompt_len:]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


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
        empty_result = EvictedGenerationResult(text="", cache_summary={})
        return empty_result if return_details else empty_result.text

    device = model_device(model)
    from transformers.cache_utils import DynamicCache

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
        past_key_values = DynamicCache()
        num_layers = k_cache.shape[0]
        for i in range(num_layers):
            past_key_values.update(k_cache[i].to(device), v_cache[i].to(device), layer_idx=i)

        input_ids = torch.tensor([[first_new_token_id]], dtype=torch.long, device=device)
        cache_len = k_cache.shape[-2]
        attention_mask = torch.ones(1, cache_len + 1, dtype=torch.long, device=device)
        position_ids = torch.tensor([[original_sequence_length]], dtype=torch.long, device=device)

        generated_tokens = [first_new_token_id]
        model_forward_parameters = inspect.signature(model.forward).parameters
        
        # We already generated the first token from the prefill step, so we need max_new_tokens - 1 more
        for _ in range(max_new_tokens - 1):
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            if "cache_position" in model_forward_parameters:
                forward_kwargs["cache_position"] = position_ids.reshape(-1)
            with torch.no_grad():
                outputs = model(**forward_kwargs)

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
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            
            if next_token.item() == tokenizer.eos_token_id:
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
        result = EvictedGenerationResult(text=text, cache_summary=cache_summary)
        return result if return_details else result.text


__all__ = [
    "HfModelBundle",
    "HfPrefillRecord",
    "EvictedGenerationResult",
    "extended_rotary_position_capacity",
    "extract_attention_obs",
    "extract_layer_kv_cache",
    "generate_text",
    "generate_text_with_evicted_cache",
    "load_hf_model_and_tokenizer",
    "model_device",
    "resolve_device",
    "resolve_torch_dtype",
    "run_hf_prefill",
]
