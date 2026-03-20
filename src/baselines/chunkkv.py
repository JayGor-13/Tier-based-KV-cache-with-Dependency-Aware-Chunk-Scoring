"""ChunkKV baseline aligned with kvpress chunk-wise selection behavior."""

from __future__ import annotations

from typing import Sequence

import torch

from src.baselines._utils import build_result_from_keep_mask
from src.core.evictor import EvictionResult

Chunk = Sequence[int] | torch.Tensor


def chunkkv_chunk_keep_indices(
    chunk_scores: torch.Tensor,
    chunks: Sequence[Chunk],
    *,
    budget: int,
    sequence_length: int | None = None,
) -> torch.Tensor:
    """Select top chunks to keep, following kvpress ChunkKV selection.

    kvpress ChunkKV computes token scores then selects chunks by chunk-level score.
    In this project we receive pre-aggregated chunk scores, so this function directly
    performs the chunk selection stage.
    """
    if chunk_scores.ndim != 1:
        raise ValueError("chunk_scores must be 1D.")
    if chunk_scores.numel() != len(chunks):
        raise ValueError("chunk_scores length must match number of chunks.")
    if len(chunks) == 0:
        return torch.empty(0, dtype=torch.long, device=chunk_scores.device)

    if sequence_length is None:
        sequence_length = 0
        for chunk in chunks:
            idx = torch.as_tensor(chunk, dtype=torch.long)
            if idx.numel() > 0:
                sequence_length = max(sequence_length, int(idx.max().item()) + 1)
    if sequence_length <= 0:
        return torch.empty(0, dtype=torch.long, device=chunk_scores.device)

    keep_ratio = max(0.0, min(1.0, float(budget) / float(sequence_length)))
    n_chunks_kept = max(1, int(round(len(chunks) * keep_ratio)))
    n_chunks_kept = min(n_chunks_kept, len(chunks))

    top_chunks = torch.topk(chunk_scores.to(torch.float32), k=n_chunks_kept, largest=True).indices
    return torch.sort(top_chunks).values


def evict_chunkkv(
    chunk_scores: torch.Tensor,
    chunks: Sequence[Chunk],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    budget: int,
    theta: float = 0.3,
    recent_window: int = 16,
    sequence_length: int | None = None,
    sink_token_index: int = 0,
) -> EvictionResult:
    """Apply ChunkKV baseline eviction.

    Notes
    -----
    `theta`, `recent_window`, and `sink_token_index` are retained for interface
    compatibility with existing scripts but are not used by ChunkKV selection.
    """
    del theta, recent_window, sink_token_index

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

    kept_chunk_ids = chunkkv_chunk_keep_indices(
        chunk_scores=chunk_scores,
        chunks=chunks,
        budget=budget,
        sequence_length=sequence_length,
    )

    keep_mask = torch.zeros(t, dtype=torch.bool, device=k_cache.device)
    for chunk_id in kept_chunk_ids.tolist():
        idx = torch.as_tensor(chunks[chunk_id], dtype=torch.long, device=k_cache.device)
        if idx.numel() == 0:
            continue
        keep_mask[idx] = True

    return build_result_from_keep_mask(
        keep_mask=keep_mask, k_cache=k_cache, v_cache=v_cache, budget=budget
    )


__all__ = ["chunkkv_chunk_keep_indices", "evict_chunkkv"]
