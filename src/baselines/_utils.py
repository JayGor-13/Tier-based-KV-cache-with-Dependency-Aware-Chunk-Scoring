"""Shared helpers for baseline eviction policies."""

from __future__ import annotations

import torch

from src.core.evictor import (
    EvictionResult,
    compute_budget_status,
    select_cache_positions,
)


def forced_keep_mask(
    *,
    sequence_length: int,
    recent_window: int,
    sink_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    if sequence_length == 0:
        return mask

    if sink_tokens > 0:
        sink_upto = min(sink_tokens, sequence_length)
        mask[:sink_upto] = True

    if recent_window > 0:
        recent_start = max(sequence_length - recent_window, 0)
        mask[recent_start:] = True

    return mask


def topk_from_candidates(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Return `k` token indices among candidates with largest score."""
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return candidates

    k = min(k, int(candidates.numel()))
    local_scores = scores[candidates]
    top_local = torch.topk(local_scores, k=k, largest=True).indices
    return candidates[top_local]


def trim_keep_mask_to_budget(
    keep_mask: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Drop the lowest-scored kept tokens until the mask respects budget."""
    target_keep = max(0, min(int(budget), int(keep_mask.numel())))
    current_keep = int(keep_mask.sum().item())
    if current_keep <= target_keep:
        return keep_mask

    kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    local_scores = scores.to(device=keep_mask.device, dtype=torch.float32)[kept_indices]
    drop_count = current_keep - target_keep
    drop_local = torch.topk(local_scores, k=drop_count, largest=False).indices
    keep_mask[kept_indices[drop_local]] = False
    return keep_mask


def resize_keep_mask_to_budget(
    keep_mask: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Resize a keep mask to the exact usable budget using token scores."""
    keep_mask = trim_keep_mask_to_budget(keep_mask, scores, budget)
    target_keep = max(0, min(int(budget), int(keep_mask.numel())))
    current_keep = int(keep_mask.sum().item())
    if current_keep >= target_keep:
        return keep_mask

    add_count = target_keep - current_keep
    add_indices = topk_from_candidates(scores, ~keep_mask, add_count)
    keep_mask[add_indices] = True
    return keep_mask


def build_result_from_keep_mask(
    *,
    keep_mask: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    budget: int,
    partially_evicted_chunks: int = 0,
) -> EvictionResult:
    kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    removed_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    new_k_cache = select_cache_positions(k_cache, kept_indices)
    new_v_cache = select_cache_positions(v_cache, kept_indices)

    target_deficit = max(int(k_cache.shape[-2]) - budget, 0)
    tokens_removed = int(removed_indices.numel())
    budget_status = compute_budget_status(
        sequence_length=int(k_cache.shape[-2]),
        budget=budget,
        kept_tokens=int(kept_indices.numel()),
    )
    return EvictionResult(
        new_k_cache=new_k_cache,
        new_v_cache=new_v_cache,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        keep_mask=keep_mask,
        target_deficit=target_deficit,
        tokens_removed=tokens_removed,
        budget_shortfall=budget_status.shortfall,
        budget_overflow=budget_status.overflow,
        budget_utilization=budget_status.utilization,
        partially_evicted_chunks=int(partially_evicted_chunks),
    )
