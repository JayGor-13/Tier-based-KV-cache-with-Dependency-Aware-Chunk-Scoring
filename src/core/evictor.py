"""Module 4: Priority-respecting eviction engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

Chunk = Sequence[int] | torch.Tensor


@dataclass(frozen=True)
class EvictionResult:
    """Structured result after KV eviction."""

    new_k_cache: torch.Tensor
    new_v_cache: torch.Tensor
    kept_indices: torch.Tensor
    removed_indices: torch.Tensor
    keep_mask: torch.Tensor
    target_deficit: int
    tokens_removed: int


def _to_index_tensor(chunk: Chunk, *, device: torch.device) -> torch.Tensor:
    if isinstance(chunk, torch.Tensor):
        idx = chunk.to(device=device, dtype=torch.long)
    else:
        idx = torch.as_tensor(chunk, dtype=torch.long, device=device)
    if idx.ndim != 1:
        raise ValueError("Each chunk must be 1D.")
    return idx


def _validate_token_indices(indices: torch.Tensor, sequence_length: int) -> None:
    if indices.numel() == 0:
        return
    min_idx = int(indices.min().item())
    max_idx = int(indices.max().item())
    if min_idx < 0 or max_idx >= sequence_length:
        raise IndexError(
            f"Chunk token indices out of range: found [{min_idx}, {max_idx}] "
            f"for sequence length {sequence_length}."
        )


def _remove_chunks_by_priority(
    *,
    keep_mask: torch.Tensor,
    chunk_scores: torch.Tensor,
    mask_tiers: torch.Tensor,
    chunks: Sequence[Chunk],
    target_tier: int,
    deficit: int,
    removed_so_far: int,
) -> int:
    if removed_so_far >= deficit:
        return removed_so_far

    candidate_chunks = torch.nonzero(mask_tiers == target_tier, as_tuple=False).flatten()
    if candidate_chunks.numel() == 0:
        return removed_so_far

    candidate_scores = chunk_scores[candidate_chunks]
    sort_order = torch.argsort(candidate_scores, descending=False)

    for local_idx in sort_order:
        chunk_idx = int(candidate_chunks[int(local_idx.item())].item())
        token_idx = _to_index_tensor(chunks[chunk_idx], device=keep_mask.device)
        _validate_token_indices(token_idx, keep_mask.numel())

        active_before = keep_mask[token_idx]
        removed_now = int(active_before.sum().item())
        keep_mask[token_idx] = False
        removed_so_far += removed_now

        if removed_so_far >= deficit:
            break

    return removed_so_far


def compute_keep_mask(
    mask_tiers: torch.Tensor,
    chunk_scores: torch.Tensor,
    chunks: Sequence[Chunk],
    *,
    sequence_length: int,
    budget: int,
    allow_level2_fallback: bool = False,
) -> tuple[torch.Tensor, int]:
    """Compute boolean keep mask according to tier-priority eviction rules."""
    if mask_tiers.ndim != 1:
        raise ValueError("`mask_tiers` must be 1D.")
    if chunk_scores.ndim != 1:
        raise ValueError("`chunk_scores` must be 1D.")
    if mask_tiers.numel() != chunk_scores.numel() or mask_tiers.numel() != len(chunks):
        raise ValueError("Mismatch among `mask_tiers`, `chunk_scores`, and `chunks`.")
    if budget < 0:
        raise ValueError("`budget` must be non-negative.")
    if sequence_length < 0:
        raise ValueError("`sequence_length` must be non-negative.")

    keep_mask = torch.ones(sequence_length, dtype=torch.bool, device=chunk_scores.device)
    if sequence_length <= budget:
        return keep_mask, 0

    deficit = sequence_length - budget
    tokens_removed = 0

    tokens_removed = _remove_chunks_by_priority(
        keep_mask=keep_mask,
        chunk_scores=chunk_scores,
        mask_tiers=mask_tiers,
        chunks=chunks,
        target_tier=0,
        deficit=deficit,
        removed_so_far=tokens_removed,
    )

    if tokens_removed < deficit:
        tokens_removed = _remove_chunks_by_priority(
            keep_mask=keep_mask,
            chunk_scores=chunk_scores,
            mask_tiers=mask_tiers,
            chunks=chunks,
            target_tier=1,
            deficit=deficit,
            removed_so_far=tokens_removed,
        )

    if tokens_removed < deficit and allow_level2_fallback:
        tokens_removed = _remove_chunks_by_priority(
            keep_mask=keep_mask,
            chunk_scores=chunk_scores,
            mask_tiers=mask_tiers,
            chunks=chunks,
            target_tier=2,
            deficit=deficit,
            removed_so_far=tokens_removed,
        )

    if tokens_removed < deficit:
        raise RuntimeError(
            "Unable to satisfy budget using Tier-0 and Tier-1 chunks only. "
            "Consider reducing protected tokens or enabling level-2 fallback."
        )

    return keep_mask, tokens_removed


def select_cache_positions(cache: torch.Tensor, kept_indices: torch.Tensor) -> torch.Tensor:
    """Select retained sequence positions along the sequence axis (-2)."""
    if cache.ndim < 2:
        raise ValueError("Cache tensor must have at least 2 dimensions.")
    idx = kept_indices.to(device=cache.device, dtype=torch.long)
    return torch.index_select(cache, dim=-2, index=idx)


def evict_kv_cache(
    mask_tiers: torch.Tensor,
    chunk_scores: torch.Tensor,
    chunks: Sequence[Chunk],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    budget: int,
    sequence_length: int | None = None,
    allow_level2_fallback: bool = False,
) -> EvictionResult:
    """Apply tier-priority eviction and return compacted KV cache tensors."""
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache and v_cache shapes must match. Got {k_cache.shape} vs {v_cache.shape}."
        )
    if k_cache.ndim < 2:
        raise ValueError("k_cache/v_cache must have at least 2 dimensions.")

    t_from_cache = int(k_cache.shape[-2])
    if sequence_length is None:
        sequence_length = t_from_cache
    if sequence_length != t_from_cache:
        raise ValueError(
            f"sequence_length ({sequence_length}) does not match cache sequence axis "
            f"({t_from_cache})."
        )

    keep_mask, tokens_removed = compute_keep_mask(
        mask_tiers=mask_tiers.to(device=k_cache.device),
        chunk_scores=chunk_scores.to(device=k_cache.device),
        chunks=chunks,
        sequence_length=sequence_length,
        budget=budget,
        allow_level2_fallback=allow_level2_fallback,
    )

    kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    removed_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()

    new_k_cache = select_cache_positions(k_cache, kept_indices)
    new_v_cache = select_cache_positions(v_cache, kept_indices)

    return EvictionResult(
        new_k_cache=new_k_cache,
        new_v_cache=new_v_cache,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        keep_mask=keep_mask,
        target_deficit=max(sequence_length - budget, 0),
        tokens_removed=tokens_removed,
    )


__all__ = [
    "EvictionResult",
    "compute_keep_mask",
    "evict_kv_cache",
    "select_cache_positions",
]
