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
    next_token_id: int | None = None

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
    k_cache, v_cache = extract_full_kv_cache(
        outputs.past_key_values,
        offload_to_cpu=offload_to_cpu,
    )
    
    next_token_id = None
    if hasattr(outputs, "logits") and outputs.logits is not None:
        next_token_id = int(torch.argmax(outputs.logits[0, -1, :]).item())

    return HfPrefillRecord(
        sample_id=sample_id,
        prompt=prompt,
        input_ids=input_ids_cpu,
        chunks=chunks,
        chunk_map=chunk_map,
        attention_obs=attention_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        next_token_id=next_token_id,
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


def generate_text_with_evicted_cache(
    *,
    model: Any,
    tokenizer: Any,
    first_new_token_id: int,
    max_new_tokens: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    original_sequence_length: int,
) -> str:
    """Generate text dynamically using an already evicted KV cache."""
    if max_new_tokens <= 0:
        return ""

    device = model_device(model)
    import inspect
    from transformers.cache_utils import DynamicCache
    
    # HF models with RoPE (like Qwen2) often do: cos, sin = rotary_emb(..., seq_len=kv_seq_len)
    # and then rotary_emb returns cos[:kv_seq_len].
    # But for evicted caches, position_ids can be larger than kv_seq_len,
    # causing an IndexError when apply_rotary_pos_emb does cos[position_ids].
    # We patch all rotary embedding modules to use the maximum needed sequence length.
    original_forwards = {}
    for name, module in model.named_modules():
        if "RotaryEmbedding" in module.__class__.__name__:
            original_forward = module.forward
            if "seq_len" not in inspect.signature(original_forward).parameters:
                continue

            original_forwards[name] = original_forward
            def make_patched_forward(orig_forward):
                def patched_forward(self, x, *args, seq_len=None, **kwargs):
                    # Force seq_len to be large enough for our position_ids
                    # We add max_new_tokens to ensure it's large enough for the whole generation
                    target_seq_len = original_sequence_length + max_new_tokens
                    if seq_len is not None and seq_len < target_seq_len:
                        seq_len = target_seq_len
                    # orig_forward is a bound method, so don't pass self
                    if seq_len is None:
                        return orig_forward(x, *args, **kwargs)
                    return orig_forward(x, *args, seq_len=seq_len, **kwargs)
                return patched_forward
            module.forward = make_patched_forward(original_forward).__get__(
                module, module.__class__
            )

    try:
        past_key_values = DynamicCache()
        num_layers = k_cache.shape[0]
        for i in range(num_layers):
            past_key_values.update(k_cache[i].to(device), v_cache[i].to(device), layer_idx=i)

        input_ids = torch.tensor([[first_new_token_id]], dtype=torch.long, device=device)
        cache_len = k_cache.shape[-2]
        attention_mask = torch.ones(1, cache_len + 1, dtype=torch.long, device=device)
        position_ids = torch.tensor([[original_sequence_length]], dtype=torch.long, device=device)

        generated_tokens = [first_new_token_id]
        
        # We already generated the first token from the prefill step, so we need max_new_tokens - 1 more
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            input_ids = next_token
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
            position_ids = position_ids + 1
            past_key_values = outputs.past_key_values

        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    finally:
        for name, orig_forward in original_forwards.items():
            module = dict(model.named_modules())[name]
            module.forward = orig_forward


__all__ = [
    "HfModelBundle",
    "HfPrefillRecord",
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
