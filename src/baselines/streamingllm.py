"""StreamingLLM baseline: Keeps attention sink (initial tokens) and recent tokens."""

from __future__ import annotations

import torch

from src.baselines._utils import build_result_from_keep_mask
from src.core.evictor import EvictionResult


def evict_streamingllm(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    budget: int,
    num_sink_tokens: int = 4,
) -> EvictionResult:
    """Apply StreamingLLM baseline eviction.
    
    Keeps `num_sink_tokens` from the beginning, and `budget - num_sink_tokens` from the end.
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
    
    # Keep sink tokens
    actual_sink_tokens = min(num_sink_tokens, budget)
    if actual_sink_tokens > 0:
        keep_mask[:actual_sink_tokens] = True
        
    # Keep recent tokens
    recent_tokens_to_keep = budget - actual_sink_tokens
    if recent_tokens_to_keep > 0:
        keep_mask[-recent_tokens_to_keep:] = True

    return build_result_from_keep_mask(
        keep_mask=keep_mask, k_cache=k_cache, v_cache=v_cache, budget=budget
    )

__all__ = ["evict_streamingllm"]
