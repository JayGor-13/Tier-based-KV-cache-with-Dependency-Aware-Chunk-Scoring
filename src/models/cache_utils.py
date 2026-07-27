"""HuggingFace model utilities for running the TDC-KV pipeline.

The core TDC-KV modules operate on token ids, observed attention, and KV-cache
tensors. This file provides the thin adapter layer from standard
`transformers` causal language models to those tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.core.chunker import SentenceBoundaryChunkConstructor


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

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.numel())


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
        model_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if "attn_implementation" not in model_kwargs:
            raise
        model_kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

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
    chunk_constructor: SentenceBoundaryChunkConstructor | None = None,
    offload_to_cpu: bool = True,
) -> HfPrefillRecord:
    """Run one prompt through a HF model and return TDC-KV-ready tensors."""
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

    with torch.no_grad():
        outputs = model(
            **encoded,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    input_ids_cpu = input_ids[0].detach().cpu()
    if chunk_constructor is None:
        chunk_constructor = SentenceBoundaryChunkConstructor(
            tokenizer=tokenizer, min_chunk_tokens=min_chunk_tokens, device="cpu"
        )
    chunks, chunk_map = chunk_constructor.forward(input_ids_cpu)
    attention_obs = extract_attention_obs(
        outputs.attentions,
        window_size=observation_window,
        mode=attention_mode,
        layer_index=layer_index,
        offload_to_cpu=offload_to_cpu,
    )
    k_cache, v_cache = extract_layer_kv_cache(
        outputs.past_key_values,
        layer_index=layer_index,
        offload_to_cpu=offload_to_cpu,
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


__all__ = [
    "HfModelBundle",
    "HfPrefillRecord",
    "extract_attention_obs",
    "extract_layer_kv_cache",
    "generate_text",
    "load_hf_model_and_tokenizer",
    "model_device",
    "resolve_device",
    "resolve_torch_dtype",
    "run_hf_prefill",
]
