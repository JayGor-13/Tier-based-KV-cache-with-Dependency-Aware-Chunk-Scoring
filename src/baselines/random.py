"""Random baseline: Randomly evicts tokens (or chunks)."""

from __future__ import annotations

from typing import Sequence

import torch

from src.baselines._utils import build_result_from_keep_mask
from src.core.evictor import EvictionResult

Chunk = Sequence[int] | torch.Tensor


def evict_random(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    budget: int,
    chunks: Sequence[Chunk] | None = None,
) -> EvictionResult:
    """Apply Random baseline eviction.
    
    If chunks are provided, evicts at the chunk level. Otherwise, evicts at the token level.
    """
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shape.")
    if k_cache.ndim < 2:
        raise ValueError("k_cache/v_cache must have at least 2 dimensions.")

    t = int(k_cache.shape[-2])
    if t <= budget:
        keep_all = torch.ones(t, dtype=torch.bool, device=k_cache.device)
        return build_result_from_keep_mask(
            keep_mask=keep_all, k_cache=k_cache, v_cache=v_cache, budget=budget
        )

    keep_mask = torch.zeros(t, dtype=torch.bool, device=k_cache.device)
    
    if chunks is None:
        # Token-level random
        indices = torch.randperm(t, device=k_cache.device)[:budget]
        keep_mask[indices] = True
    else:
        # Chunk-level random
        # We need to pick random chunks until budget is filled.
        # To avoid exceeding budget, we might pick slightly less.
        chunk_indices = torch.randperm(len(chunks), device=k_cache.device)
        current_tokens = 0
        for chunk_id in chunk_indices.tolist():
            idx = torch.as_tensor(chunks[chunk_id], dtype=torch.long, device=k_cache.device)
            if idx.numel() == 0:
                continue
            if current_tokens + idx.numel() > budget and current_tokens > 0:
                # Stop if it exceeds budget, but allow at least one chunk
                break
            keep_mask[idx] = True
            current_tokens += idx.numel()

    return build_result_from_keep_mask(
        keep_mask=keep_mask, k_cache=k_cache, v_cache=v_cache, budget=budget
    )

__all__ = ["evict_random"]
