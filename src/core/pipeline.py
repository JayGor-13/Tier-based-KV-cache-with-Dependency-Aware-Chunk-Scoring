"""End-to-end orchestration for the 4-module TDC-KV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from src.core.chunker import MIN_CHUNK_TOKENS, SentenceBoundaryChunkConstructor
from src.core.evictor import EvictionResult, evict_kv_cache
from src.core.masker import assign_protection_tiers
from src.core.scorer import DualSignalScorer


@dataclass(frozen=True)
class TDCKVPipelineResult:
    """Structured outputs from one TDC-KV eviction pass."""

    chunks: list[Tensor]
    chunk_map: Tensor
    chunk_scores: Tensor
    mask_tiers: Tensor
    eviction: EvictionResult


class TDCKVPipeline:
    """Compose Module 1 -> Module 2 -> Module 3 -> Module 4."""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        punct_ids: set[int] | None = None,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
        alpha: float = 0.6,
        beta: float = 0.4,
        window_size: int = 16,
        num_layers: int | None = None,
        theta: float = 0.3,
        recent_window: int = 16,
        sink_token_index: int = 0,
        allow_level2_fallback: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.theta = float(theta)
        self.recent_window = int(recent_window)
        self.sink_token_index = int(sink_token_index)
        self.allow_level2_fallback = bool(allow_level2_fallback)

        self.chunker = SentenceBoundaryChunkConstructor(
            tokenizer=tokenizer,
            punct_ids=punct_ids,
            min_chunk_tokens=min_chunk_tokens,
            device=self.device,
        )
        self.scorer = DualSignalScorer(
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            num_layers=num_layers,
            device=self.device,
        )

    def run(
        self,
        *,
        token_ids: Iterable[int] | Tensor,
        attention_obs: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        budget: int,
        sequence_length: int | None = None,
        theta: float | None = None,
        recent_window: int | None = None,
        sink_token_index: int | None = None,
        allow_level2_fallback: bool | None = None,
    ) -> TDCKVPipelineResult:
        """Run one full eviction pass from raw token IDs and observed attention."""
        if isinstance(token_ids, Tensor):
            token_tensor = token_ids.to(device=self.device, dtype=torch.long)
        else:
            token_tensor = torch.as_tensor(
                list(token_ids), dtype=torch.long, device=self.device
            )

        chunks, chunk_map = self.chunker.forward(token_tensor)
        chunk_scores = self.scorer.forward(attention_obs.to(self.device), chunks)

        t = int(k_cache.shape[-2]) if sequence_length is None else int(sequence_length)
        run_theta = self.theta if theta is None else float(theta)
        run_recent_window = (
            self.recent_window if recent_window is None else int(recent_window)
        )
        run_sink_token_index = (
            self.sink_token_index
            if sink_token_index is None
            else int(sink_token_index)
        )
        run_allow_level2_fallback = (
            self.allow_level2_fallback
            if allow_level2_fallback is None
            else bool(allow_level2_fallback)
        )

        tiers = assign_protection_tiers(
            chunk_scores=chunk_scores,
            chunks=chunks,
            theta=run_theta,
            recent_window=run_recent_window,
            sequence_length=t,
            sink_token_index=run_sink_token_index,
        )
        eviction = evict_kv_cache(
            mask_tiers=tiers,
            chunk_scores=chunk_scores,
            chunks=chunks,
            k_cache=k_cache,
            v_cache=v_cache,
            budget=int(budget),
            sequence_length=t,
            allow_level2_fallback=run_allow_level2_fallback,
        )

        return TDCKVPipelineResult(
            chunks=chunks,
            chunk_map=chunk_map,
            chunk_scores=chunk_scores,
            mask_tiers=tiers,
            eviction=eviction,
        )


def build_tdc_kv_pipeline(
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    punct_ids: set[int] | None = None,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    alpha: float = 0.6,
    beta: float = 0.4,
    window_size: int = 16,
    num_layers: int | None = None,
    theta: float = 0.3,
    recent_window: int = 16,
    sink_token_index: int = 0,
    allow_level2_fallback: bool = False,
    device: str | torch.device = "cpu",
) -> TDCKVPipeline:
    """Factory helper for constructing the end-to-end TDC-KV pipeline."""
    return TDCKVPipeline(
        tokenizer=tokenizer,
        punct_ids=punct_ids,
        min_chunk_tokens=min_chunk_tokens,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
        num_layers=num_layers,
        theta=theta,
        recent_window=recent_window,
        sink_token_index=sink_token_index,
        allow_level2_fallback=allow_level2_fallback,
        device=device,
    )


__all__ = ["TDCKVPipeline", "TDCKVPipelineResult", "build_tdc_kv_pipeline"]
