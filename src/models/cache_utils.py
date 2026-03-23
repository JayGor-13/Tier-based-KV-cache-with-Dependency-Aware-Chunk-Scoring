"""Utilities for integrating TDC-KV with Hugging Face `past_key_values`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor

from src.core.evictor import compute_keep_mask


LegacyPastKeyValues = tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class PastKeyValueEvictionResult:
    """Result of cache eviction over all transformer layers."""

    new_past_key_values: Any
    kept_indices: Tensor
    removed_indices: Tensor
    keep_mask: Tensor
    target_deficit: int
    tokens_removed: int


def _legacy_cache_adapter(
    past_key_values: Any,
) -> tuple[LegacyPastKeyValues, Callable[[LegacyPastKeyValues], Any]]:
    """Convert cache container to legacy tuple format and provide restorer."""
    if past_key_values is None:
        raise ValueError("`past_key_values` cannot be None.")

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = tuple(tuple(layer) for layer in past_key_values.to_legacy_cache())

        def restore(new_legacy: LegacyPastKeyValues) -> Any:
            from_legacy = getattr(type(past_key_values), "from_legacy_cache", None)
            if callable(from_legacy):
                return from_legacy(new_legacy)
            return new_legacy

        return legacy, restore

    if isinstance(past_key_values, (tuple, list)):
        legacy = tuple(tuple(layer) for layer in past_key_values)
        return legacy, lambda new_legacy: new_legacy

    raise TypeError(
        "Unsupported past_key_values type. Expected tuple/list or cache object "
        "with `to_legacy_cache()`."
    )


def cache_sequence_length(past_key_values: Any) -> int:
    """Read sequence length from first layer key cache."""
    legacy, _ = _legacy_cache_adapter(past_key_values)
    if len(legacy) == 0:
        return 0
    first_key = legacy[0][0]
    if not isinstance(first_key, torch.Tensor):
        raise TypeError("Expected tensor key cache in first layer.")
    return int(first_key.shape[-2])


def first_layer_cache_tensors(past_key_values: Any) -> tuple[Tensor, Tensor]:
    """Return `(k_cache, v_cache)` tensors from the first cache layer."""
    legacy, _ = _legacy_cache_adapter(past_key_values)
    if len(legacy) == 0:
        raise ValueError("Empty `past_key_values`.")
    layer0 = legacy[0]
    if len(layer0) < 2:
        raise ValueError("Each cache layer must include at least key and value tensors.")
    k_cache = layer0[0]
    v_cache = layer0[1]
    if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
        raise TypeError("Key and value cache entries must be tensors.")
    return k_cache, v_cache


def select_cache_positions(cache: Tensor, kept_indices: Tensor) -> Tensor:
    """Select retained sequence positions along sequence axis `-2`."""
    if cache.ndim < 2:
        raise ValueError("Cache tensor must have at least 2 dimensions.")
    idx = kept_indices.to(device=cache.device, dtype=torch.long)
    return torch.index_select(cache, dim=-2, index=idx)


def apply_keep_indices_to_past_key_values(
    past_key_values: Any,
    kept_indices: Tensor,
) -> Any:
    """Apply same token retention indices to every layer in `past_key_values`."""
    legacy, restore = _legacy_cache_adapter(past_key_values)
    new_layers: list[tuple[Any, ...]] = []
    for layer in legacy:
        if len(layer) < 2:
            raise ValueError(
                "Each cache layer must include at least key and value tensors."
            )
        key = layer[0]
        value = layer[1]
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("Cache key/value entries must be tensors.")
        new_key = select_cache_positions(key, kept_indices)
        new_value = select_cache_positions(value, kept_indices)
        if len(layer) == 2:
            new_layers.append((new_key, new_value))
        else:
            new_layers.append((new_key, new_value, *layer[2:]))
    return restore(tuple(new_layers))


def compute_eviction_indices(
    *,
    mask_tiers: Tensor,
    chunk_scores: Tensor,
    chunks: Sequence[Sequence[int] | Tensor],
    sequence_length: int,
    budget: int,
    allow_level2_fallback: bool = False,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Compute keep/remove indices once using Module 4 tier-priority policy."""
    keep_mask, tokens_removed = compute_keep_mask(
        mask_tiers=mask_tiers,
        chunk_scores=chunk_scores,
        chunks=chunks,
        sequence_length=sequence_length,
        budget=budget,
        allow_level2_fallback=allow_level2_fallback,
    )
    kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    removed_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    return keep_mask, kept_indices, removed_indices, tokens_removed


def evict_past_key_values(
    *,
    past_key_values: Any,
    mask_tiers: Tensor,
    chunk_scores: Tensor,
    chunks: Sequence[Sequence[int] | Tensor],
    budget: int,
    allow_level2_fallback: bool = False,
) -> PastKeyValueEvictionResult:
    """Evict cached sequence positions for all layers using one keep mask."""
    sequence_length = cache_sequence_length(past_key_values)
    keep_mask, kept_indices, removed_indices, tokens_removed = compute_eviction_indices(
        mask_tiers=mask_tiers,
        chunk_scores=chunk_scores,
        chunks=chunks,
        sequence_length=sequence_length,
        budget=budget,
        allow_level2_fallback=allow_level2_fallback,
    )
    new_past_key_values = apply_keep_indices_to_past_key_values(
        past_key_values, kept_indices
    )
    return PastKeyValueEvictionResult(
        new_past_key_values=new_past_key_values,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        keep_mask=keep_mask,
        target_deficit=max(sequence_length - budget, 0),
        tokens_removed=tokens_removed,
    )


def stack_observed_attention(
    attentions: Sequence[Tensor] | Tensor,
    *,
    window_size: int,
) -> Tensor:
    """Convert HF attention outputs to `[L, H, w, t]` tensor."""
    if isinstance(attentions, torch.Tensor):
        if attentions.ndim == 3:
            # [H, w, t] -> add layer dim
            return attentions.unsqueeze(0).to(torch.float32)
        if attentions.ndim == 4:
            # [L, H, w, t]
            return attentions.to(torch.float32)
        raise ValueError(
            "Tensor attentions must have shape [H,w,t] or [L,H,w,t]. "
            f"Got {tuple(attentions.shape)}."
        )

    if len(attentions) == 0:
        raise ValueError("Empty attention sequence.")

    layers: list[Tensor] = []
    for layer_attn in attentions:
        if layer_attn.ndim != 4:
            raise ValueError(
                "Expected per-layer attention of shape [batch, heads, q, k]. "
                f"Got {tuple(layer_attn.shape)}."
            )
        # Single-sample prefill/generation integration path.
        if layer_attn.shape[0] != 1:
            raise ValueError(
                "Only batch_size=1 is supported in this helper for now."
            )
        q_len = int(layer_attn.shape[-2])
        win = max(1, min(int(window_size), q_len))
        layers.append(layer_attn[0, :, -win:, :].to(torch.float32))
    return torch.stack(layers, dim=0)


def last_layer_observed_attention(
    attentions: Sequence[Tensor] | Tensor,
    *,
    window_size: int,
) -> Tensor:
    """Extract last-layer observed attention as `[H, w, t]`."""
    stacked = stack_observed_attention(attentions, window_size=window_size)
    return stacked[-1]


def build_chunk_token_scores(
    chunk_scores: Tensor,
    chunks: Sequence[Sequence[int] | Tensor],
    *,
    sequence_length: int,
    device: torch.device | None = None,
) -> Tensor:
    """Expand chunk-level scores to token-level scores."""
    out = torch.zeros(sequence_length, dtype=torch.float32, device=device)
    for chunk_idx, chunk in enumerate(chunks):
        idx = torch.as_tensor(chunk, dtype=torch.long, device=device)
        if idx.numel() == 0:
            continue
        out[idx] = float(chunk_scores[chunk_idx].item())
    return out


__all__ = [
    "PastKeyValueEvictionResult",
    "apply_keep_indices_to_past_key_values",
    "build_chunk_token_scores",
    "cache_sequence_length",
    "compute_eviction_indices",
    "evict_past_key_values",
    "first_layer_cache_tensors",
    "last_layer_observed_attention",
    "select_cache_positions",
    "stack_observed_attention",
]
