"""H2O-style baseline using observed-attention heavy hitters + recency."""

from __future__ import annotations

import torch

from src.baselines._utils import build_result_from_keep_mask, forced_keep_mask, topk_from_candidates
from src.core.evictor import EvictionResult


def h2o_token_scores(attention_obs: torch.Tensor) -> torch.Tensor:
    """Compute observed-attention scores (kvpress ObservedAttention style).

    kvpress ObservedAttentionPress computes token score from summed attention and
    normalizes by number of contributing queries. This function mirrors that idea
    for a local `attention_obs` tensor with shape [H, w, t].
    """
    if attention_obs.ndim != 3:
        raise ValueError("attention_obs must have shape [H, w, t].")
    n_heads, window_size, seq_len = attention_obs.shape
    if seq_len == 0:
        return torch.empty(0, dtype=torch.float32, device=attention_obs.device)

    scores = attention_obs.to(torch.float32).sum(dim=1)  # [H, t]

    # Approximate kvpress causal normalization for observed-window queries:
    # for old keys, all `window_size` queries can attend; for tail keys it decreases.
    n_tokens_in_sum = torch.full((seq_len,), float(window_size), dtype=scores.dtype, device=scores.device)
    tail = min(window_size, seq_len)
    n_tokens_in_sum[seq_len - tail :] = torch.arange(
        tail, 0, -1, dtype=scores.dtype, device=scores.device
    )
    scores = scores / n_tokens_in_sum
    scores = scores.mean(dim=0)  # average over heads
    return scores


def evict_h2o(
    attention_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    budget: int,
    recent_window: int = 16,
    sink_tokens: int = 1,
    heavy_hitter_ratio: float = 1.0,
) -> EvictionResult:
    """Evict using H2O-style policy.

    This composes:
    - observed-attention heavy-hitter scoring (kvpress ObservedAttention-related),
    - recency/sink protection (StreamingLLM-style safety).
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

    scores = h2o_token_scores(attention_obs).to(device=k_cache.device)
    if scores.numel() != t:
        raise ValueError(
            f"attention_obs token axis ({scores.numel()}) does not match cache length ({t})."
        )

    keep_mask = forced_keep_mask(
        sequence_length=t,
        recent_window=recent_window,
        sink_tokens=sink_tokens,
        device=k_cache.device,
    )

    forced_count = int(keep_mask.sum().item())
    remaining_capacity = max(min(budget, t) - forced_count, 0)
    if remaining_capacity <= 0:
        return build_result_from_keep_mask(
            keep_mask=keep_mask, k_cache=k_cache, v_cache=v_cache, budget=budget
        )

    heavy_hitter_ratio = max(0.0, min(1.0, float(heavy_hitter_ratio)))
    heavy_k = int(round(float(heavy_hitter_ratio) * float(remaining_capacity)))
    heavy_k = max(0, min(heavy_k, remaining_capacity))

    candidate_mask = ~keep_mask
    heavy_idx = topk_from_candidates(scores, candidate_mask, heavy_k)
    keep_mask[heavy_idx] = True

    fill_capacity = max(remaining_capacity - int(heavy_idx.numel()), 0)
    if fill_capacity > 0:
        filler_candidates = ~keep_mask
        filler_idx = topk_from_candidates(scores, filler_candidates, fill_capacity)
        keep_mask[filler_idx] = True

    if int(keep_mask.sum().item()) > budget:
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        keep_scores = scores[keep_idx]
        drop_n = int(keep_mask.sum().item()) - budget
        to_drop_local = torch.topk(keep_scores, k=drop_n, largest=False).indices
        keep_mask[keep_idx[to_drop_local]] = False

    return build_result_from_keep_mask(
        keep_mask=keep_mask, k_cache=k_cache, v_cache=v_cache, budget=budget
    )


__all__ = ["evict_h2o", "h2o_token_scores"]
