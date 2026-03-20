"""Shared helpers for baseline eviction policies."""

from __future__ import annotations

import torch

from src.core.evictor import EvictionResult, select_cache_positions


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


def build_result_from_keep_mask(
    *,
    keep_mask: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    budget: int,
) -> EvictionResult:
    kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    removed_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()
    new_k_cache = select_cache_positions(k_cache, kept_indices)
    new_v_cache = select_cache_positions(v_cache, kept_indices)

    target_deficit = max(int(k_cache.shape[-2]) - budget, 0)
    tokens_removed = int(removed_indices.numel())
    return EvictionResult(
        new_k_cache=new_k_cache,
        new_v_cache=new_v_cache,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        keep_mask=keep_mask,
        target_deficit=target_deficit,
        tokens_removed=tokens_removed,
    )
