"""Module 3: Protection mask assignment.

This module converts continuous chunk scores into discrete protection tiers:
`0` (eviction candidates), `1` (soft-protected), and `2` (hard-protected).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

Chunk = Sequence[int] | torch.Tensor


@dataclass(frozen=True)
class MaskerResult:
    """Detailed output for the protection masker."""

    tiers: torch.Tensor
    threshold: float
    sink_chunk_index: int
    recent_start_chunk_index: int
    tier1_source: str = "chunk_scores"


def _to_index_tensor(chunk: Chunk, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(chunk, torch.Tensor):
        idx = chunk.to(device=device, dtype=torch.long)
    else:
        idx = torch.as_tensor(chunk, dtype=torch.long, device=device)
    if idx.ndim != 1:
        raise ValueError("Each chunk must be a 1D tensor/array of token indices.")
    return idx


def infer_sequence_length(chunks: Sequence[Chunk]) -> int:
    """Infer sequence length (t) from chunk indices."""
    if not chunks:
        return 0
    max_idx = -1
    for chunk in chunks:
        idx = _to_index_tensor(chunk)
        if idx.numel() == 0:
            continue
        max_idx = max(max_idx, int(idx.max().item()))
    return max_idx + 1


def find_chunk_index(chunks: Sequence[Chunk], token_index: int) -> int:
    """Find the chunk index containing `token_index`, or -1 when absent."""
    for chunk_idx, chunk in enumerate(chunks):
        idx = _to_index_tensor(chunk)
        if idx.numel() == 0:
            continue
        if bool((idx == token_index).any().item()):
            return chunk_idx
    return -1


def assign_protection_tiers(
    chunk_scores: torch.Tensor,
    chunks: Sequence[Chunk],
    *,
    theta: float = 0.3,
    recent_window: int = 16,
    sequence_length: int | None = None,
    sink_token_index: int = 0,
    protection_scores: torch.Tensor | None = None,
    return_details: bool = False,
) -> torch.Tensor | MaskerResult:
    """Assign tier mask Pi from chunk scores and chunk definitions.

    Args:
        chunk_scores: Tensor with shape `[M]`.
        chunks: Length-`M` list where each item is a 1D list/tensor of token indices.
        theta: Top ratio for soft-protection (Tier 1). `theta=0.3` means top 30%.
        recent_window: Number of most-recent tokens hard-protected (Tier 2).
        sequence_length: Optional explicit sequence length `t`.
        sink_token_index: Token index used as attention sink (usually 0).
        protection_scores: Optional independent scores used only for Tier 1.
            TDC-KV supplies dependency-routing scores here while retaining the
            fused ``chunk_scores`` for within-tier eviction ordering.
        return_details: Return `MaskerResult` instead of only the tiers tensor.

    Returns:
        Either the tiers tensor (`int8`, shape `[M]`) or `MaskerResult`.
    """
    if chunk_scores.ndim != 1:
        raise ValueError("`chunk_scores` must be a 1D tensor of shape [M].")
    if len(chunks) != chunk_scores.numel():
        raise ValueError(
            f"Mismatch between chunks ({len(chunks)}) and chunk_scores "
            f"({chunk_scores.numel()})."
        )
    if recent_window < 0:
        raise ValueError("`recent_window` must be non-negative.")
    if not 0.0 <= float(theta) <= 1.0:
        raise ValueError("`theta` must be between 0.0 and 1.0.")
    if protection_scores is not None:
        if protection_scores.ndim != 1:
            raise ValueError("`protection_scores` must be a 1D tensor.")
        if protection_scores.numel() != chunk_scores.numel():
            raise ValueError(
                "`protection_scores` length must match `chunk_scores`."
            )

    if sequence_length is None:
        sequence_length = infer_sequence_length(chunks)
    if sequence_length < 0:
        raise ValueError("`sequence_length` must be non-negative.")

    tiers = torch.zeros_like(chunk_scores, dtype=torch.int8)

    sink_chunk_index = find_chunk_index(chunks, sink_token_index)
    if sink_chunk_index >= 0:
        tiers[sink_chunk_index] = 2

    recent_start_chunk_index = -1
    if sequence_length > 0:
        recent_start_token = max(sequence_length - recent_window, 0)
        recent_start_chunk_index = find_chunk_index(chunks, recent_start_token)
        if recent_start_chunk_index >= 0:
            tiers[recent_start_chunk_index:] = 2

    threshold = float("inf")
    if chunk_scores.numel() > 0 and theta > 0:
        tier1_scores = (
            protection_scores if protection_scores is not None else chunk_scores
        )
        clean_scores = torch.nan_to_num(
            tier1_scores.to(device=chunk_scores.device, dtype=torch.float32),
            nan=-1e9,
            neginf=-1e9,
            posinf=1e9,
        )
        eligible = torch.nonzero(tiers != 2, as_tuple=False).flatten()
        protect_count = min(
            int(eligible.numel()),
            int(math.ceil(float(theta) * float(eligible.numel()))),
        )
        if protect_count > 0:
            eligible_scores = clean_scores[eligible]
            order = torch.argsort(
                eligible_scores, descending=True, stable=True
            )
            protected = eligible[order[:protect_count]]
            tiers[protected] = 1
            threshold = float(clean_scores[protected].min().item())

    if return_details:
        return MaskerResult(
            tiers=tiers,
            threshold=threshold,
            sink_chunk_index=sink_chunk_index,
            recent_start_chunk_index=recent_start_chunk_index,
            tier1_source=(
                "disabled"
                if theta <= 0
                else (
                    "dependency_scores"
                    if protection_scores is not None
                    else "chunk_scores"
                )
            ),
        )
    return tiers


__all__ = [
    "MaskerResult",
    "assign_protection_tiers",
    "find_chunk_index",
    "infer_sequence_length",
]
