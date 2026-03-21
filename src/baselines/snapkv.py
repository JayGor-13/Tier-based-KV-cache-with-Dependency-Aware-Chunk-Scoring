"""SnapKV baseline aligned with kvpress SnapKV scoring behavior."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.baselines._utils import build_result_from_keep_mask, forced_keep_mask, topk_from_candidates
from src.core.evictor import EvictionResult


def snapkv_token_scores(
    attention_obs: torch.Tensor,
    *,
    window_size: int = 64,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Compute SnapKV token scores from observed attention.

    Mirrors kvpress SnapKVPress logic at a simplified tensor-interface level:
    1. focus on last `window_size` queries,
    2. score only pre-window tokens via mean attention,
    3. apply avg-pooling smoothing,
    4. pad observation window with max score so recent tokens are retained.
    """
    if attention_obs.ndim != 3:
        raise ValueError("attention_obs must have shape [H, w, t].")
    n_heads, n_queries, seq_len = attention_obs.shape
    if seq_len == 0:
        return torch.empty(0, dtype=torch.float32, device=attention_obs.device)

    effective_window = int(max(1, min(window_size, n_queries, seq_len)))
    pre_window_len = seq_len - effective_window

    if pre_window_len <= 0:
        # Degenerate case: entire sequence is the observation window.
        return torch.ones(seq_len, dtype=torch.float32, device=attention_obs.device)

    attn_weights = attention_obs[:, -effective_window:, :pre_window_len].to(torch.float32)
    scores = attn_weights.mean(dim=-2)  # [H, pre_window_len]

    effective_kernel = max(1, min(kernel_size, pre_window_len))
    # Match the original implementation's smoothing step.
    scores = F.avg_pool1d(
        scores.unsqueeze(0),
        kernel_size=effective_kernel,
        padding=effective_kernel // 2,
        stride=1,
    ).squeeze(0)
    scores = scores[..., :pre_window_len]  # Guard for even-kernel edge cases.

    # kvpress averages over kv-groups after reshaping; here we average over heads.
    scores = scores.mean(dim=0)  # [pre_window_len]
    max_score = float(scores.max().item()) if scores.numel() > 0 else 1.0
    return F.pad(scores, (0, effective_window), value=max_score)


def evict_snapkv(
    attention_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    budget: int,
    recent_window: int = 64,
    sink_tokens: int = 0,
    kernel_size: int = 5,
) -> EvictionResult:
    """Evict using SnapKV baseline policy.

    Uses SnapKV scores and keeps top tokens globally. Optional sink tokens are
    additionally forced to remain for compatibility with existing local pipeline.
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

    scores = snapkv_token_scores(
        attention_obs, window_size=recent_window, kernel_size=kernel_size
    ).to(device=k_cache.device)
    if scores.numel() != t:
        raise ValueError(
            f"attention_obs token axis ({scores.numel()}) does not match cache length ({t})."
        )

    keep_mask = torch.zeros(t, dtype=torch.bool, device=k_cache.device)
    n_keep = max(min(int(budget), t), 0)

    # kvpress SnapKV retains recent window via score padding. Keep sink optional.
    if sink_tokens > 0:
        sink_mask = forced_keep_mask(
            sequence_length=t,
            recent_window=0,
            sink_tokens=sink_tokens,
            device=k_cache.device,
        )
        keep_mask[sink_mask] = True

    fixed_count = int(keep_mask.sum().item())
    remaining = max(n_keep - fixed_count, 0)
    if remaining > 0:
        candidate_mask = ~keep_mask
        dynamic_keep = topk_from_candidates(scores, candidate_mask, remaining)
        keep_mask[dynamic_keep] = True

    return build_result_from_keep_mask(
        keep_mask=keep_mask, k_cache=k_cache, v_cache=v_cache, budget=budget
    )


__all__ = ["evict_snapkv", "snapkv_token_scores"]
