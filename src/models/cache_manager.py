"""Stateful bounded-cache management for autoregressive decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

Chunk = Sequence[int] | torch.Tensor


@dataclass(frozen=True)
class CacheTrimEvent:
    logical_length: int
    tokens_before: int
    tokens_after: int
    tokens_removed: int
    groups_removed: int
    used_tier2_fallback: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class DecodingCacheManager:
    """Track cache metadata and enforce a token budget after every decode step."""

    def __init__(
        self,
        *,
        budget: int,
        recent_window: int,
        logical_positions: torch.Tensor,
        group_ids: torch.Tensor,
        token_scores: torch.Tensor,
        base_tiers: torch.Tensor,
        sink_group_id: int | None,
        next_group_id: int,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        if recent_window < 0:
            raise ValueError("recent_window must be non-negative.")

        self.budget = int(budget)
        self.recent_window = int(recent_window)
        self.logical_positions = logical_positions.to(dtype=torch.long, device="cpu")
        self.group_ids = group_ids.to(dtype=torch.long, device="cpu")
        self.token_scores = token_scores.to(dtype=torch.float32, device="cpu")
        self.base_tiers = base_tiers.to(dtype=torch.int8, device="cpu")
        self.sink_group_id = sink_group_id
        self.next_group_id = int(next_group_id)
        self.events: list[CacheTrimEvent] = []
        self.initial_input_tokens = int(self.logical_positions.numel())
        self.initial_post_trim_tokens: int | None = None

        lengths = {
            int(self.logical_positions.numel()),
            int(self.group_ids.numel()),
            int(self.token_scores.numel()),
            int(self.base_tiers.numel()),
        }
        if len(lengths) != 1:
            raise ValueError("All cache metadata tensors must have equal length.")

    @classmethod
    def from_prompt(
        cls,
        *,
        budget: int,
        recent_window: int,
        kept_indices: torch.Tensor,
        original_sequence_length: int,
        chunks: Sequence[Chunk] | None = None,
        chunk_scores: torch.Tensor | None = None,
        mask_tiers: torch.Tensor | None = None,
        sink_token_index: int = 0,
    ) -> "DecodingCacheManager":
        kept = kept_indices.detach().to(dtype=torch.long, device="cpu")
        if kept.ndim != 1:
            raise ValueError("kept_indices must be a 1D tensor.")
        if kept.numel() > 0:
            low = int(kept.min().item())
            high = int(kept.max().item())
            if low < 0 or high >= original_sequence_length:
                raise IndexError("kept_indices exceed the original prompt length.")

        if chunks is None:
            group_ids = torch.arange(kept.numel(), dtype=torch.long)
            scores = torch.zeros(kept.numel(), dtype=torch.float32)
            tiers = torch.zeros(kept.numel(), dtype=torch.int8)
            sink_matches = torch.nonzero(
                kept == int(sink_token_index), as_tuple=False
            ).flatten()
            sink_group_id = (
                int(group_ids[int(sink_matches[0].item())].item())
                if sink_matches.numel() > 0
                else None
            )
            next_group_id = int(kept.numel())
        else:
            token_to_group = torch.full(
                (int(original_sequence_length),), -1, dtype=torch.long
            )
            for group_id, chunk in enumerate(chunks):
                idx = torch.as_tensor(chunk, dtype=torch.long, device="cpu")
                if idx.ndim != 1:
                    raise ValueError("Each chunk must be a 1D tensor or sequence.")
                if idx.numel() > 0:
                    token_to_group[idx] = group_id
            group_ids = token_to_group[kept]
            if bool((group_ids < 0).any().item()):
                raise ValueError("Every retained prompt token must belong to a chunk.")

            if chunk_scores is None:
                chunk_score_values = torch.zeros(len(chunks), dtype=torch.float32)
            else:
                chunk_score_values = chunk_scores.detach().to(
                    dtype=torch.float32, device="cpu"
                )
                if chunk_score_values.numel() != len(chunks):
                    raise ValueError("chunk_scores length must match chunks.")
            scores = chunk_score_values[group_ids]

            if mask_tiers is None:
                chunk_tiers = torch.zeros(len(chunks), dtype=torch.int8)
            else:
                chunk_tiers = mask_tiers.detach().to(dtype=torch.int8, device="cpu")
                if chunk_tiers.numel() != len(chunks):
                    raise ValueError("mask_tiers length must match chunks.")
            # Tier 2 recency is recomputed from logical positions at every step.
            tiers = torch.where(
                chunk_tiers[group_ids] == 1,
                torch.ones_like(group_ids, dtype=torch.int8),
                torch.zeros_like(group_ids, dtype=torch.int8),
            )
            sink_group = token_to_group[int(sink_token_index)]
            sink_group_id = int(sink_group.item()) if sink_group >= 0 else None
            next_group_id = len(chunks)

        return cls(
            budget=budget,
            recent_window=recent_window,
            logical_positions=kept,
            group_ids=group_ids,
            token_scores=scores,
            base_tiers=tiers,
            sink_group_id=sink_group_id,
            next_group_id=next_group_id,
        )

    @property
    def cache_length(self) -> int:
        return int(self.logical_positions.numel())

    def append_generated_token(
        self,
        *,
        logical_position: int,
        score: float = 0.0,
    ) -> None:
        if self.logical_positions.numel() > 0:
            latest = int(self.logical_positions.max().item())
            if logical_position <= latest:
                raise ValueError("Generated logical positions must increase monotonically.")

        self.logical_positions = torch.cat(
            [self.logical_positions, torch.tensor([logical_position], dtype=torch.long)]
        )
        self.group_ids = torch.cat(
            [self.group_ids, torch.tensor([self.next_group_id], dtype=torch.long)]
        )
        self.token_scores = torch.cat(
            [self.token_scores, torch.tensor([score], dtype=torch.float32)]
        )
        self.base_tiers = torch.cat(
            [self.base_tiers, torch.tensor([0], dtype=torch.int8)]
        )
        self.next_group_id += 1

    def trim_cache_tensors(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        current_logical_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, CacheTrimEvent]:
        if k_cache.shape != v_cache.shape:
            raise ValueError("k_cache and v_cache must have matching shapes.")
        self._validate_physical_length(int(k_cache.shape[-2]))
        kept_positions, event = self._plan_trim(current_logical_length)
        index = kept_positions.to(device=k_cache.device)
        new_k = torch.index_select(k_cache, dim=-2, index=index)
        new_v = torch.index_select(v_cache, dim=-2, index=index.to(v_cache.device))
        self._apply_trim(kept_positions, event)
        return new_k, new_v, event

    def trim_past_key_values(
        self,
        past_key_values: Any,
        *,
        current_logical_length: int,
    ) -> tuple[Any, CacheTrimEvent]:
        legacy = past_key_values
        if hasattr(legacy, "to_legacy_cache"):
            legacy = legacy.to_legacy_cache()
        elif hasattr(legacy, "key_cache") and hasattr(legacy, "value_cache"):
            legacy = list(zip(legacy.key_cache, legacy.value_cache))
        if not legacy:
            raise ValueError("past_key_values contains no cache layers.")

        first_key = legacy[0][0]
        self._validate_physical_length(int(first_key.shape[-2]))
        kept_positions, event = self._plan_trim(current_logical_length)

        from transformers.cache_utils import DynamicCache

        compacted = DynamicCache()
        for layer_index, layer in enumerate(legacy):
            key, value = layer[:2]
            index = kept_positions.to(device=key.device)
            compacted.update(
                torch.index_select(key, dim=-2, index=index),
                torch.index_select(value, dim=-2, index=index.to(value.device)),
                layer_idx=layer_index,
            )
        self._apply_trim(kept_positions, event)
        return compacted, event

    def summary(self) -> dict[str, int]:
        post_trim = [event.tokens_after for event in self.events]
        return {
            "budget": self.budget,
            "initial_input_tokens": self.initial_input_tokens,
            "initial_post_trim_tokens": (
                self.initial_post_trim_tokens
                if self.initial_post_trim_tokens is not None
                else self.cache_length
            ),
            "final_cache_tokens": self.cache_length,
            "max_post_trim_tokens": max(post_trim, default=self.cache_length),
            "trim_checks": len(self.events),
            "re_eviction_events": sum(
                1 for event in self.events if event.tokens_removed > 0
            ),
            "tier2_fallback_events": sum(
                1 for event in self.events if event.used_tier2_fallback
            ),
            "total_tokens_removed": sum(
                event.tokens_removed for event in self.events
            ),
            "budget_violations": sum(
                1 for event in self.events if event.tokens_after > self.budget
            ),
        }

    def _plan_trim(
        self, current_logical_length: int
    ) -> tuple[torch.Tensor, CacheTrimEvent]:
        tokens_before = self.cache_length
        keep_mask = torch.ones(tokens_before, dtype=torch.bool)
        removed_groups = 0
        used_tier2_fallback = False
        recent_start = max(int(current_logical_length) - self.recent_window, 0)

        while int(keep_mask.sum().item()) > self.budget:
            candidates: list[tuple[int, int, float, int, int]] = []
            active_groups = torch.unique(self.group_ids[keep_mask], sorted=True)
            for group_tensor in active_groups:
                group_id = int(group_tensor.item())
                group_mask = keep_mask & (self.group_ids == group_id)
                positions = self.logical_positions[group_mask]
                is_sink = self.sink_group_id == group_id
                is_recent = bool((positions >= recent_start).any().item())
                base_tier = int(self.base_tiers[group_mask].max().item())
                tier = 2 if is_sink or is_recent else base_tier
                score = float(self.token_scores[group_mask].mean().item())
                oldest = int(positions.min().item())
                # Lower tiers/scores are removed first. Sink groups sort last
                # within Tier 2, and older groups break remaining ties.
                candidates.append((tier, int(is_sink), score, oldest, group_id))

            if not candidates:
                break
            tier, _, _, _, selected_group = min(candidates)
            if tier == 2:
                used_tier2_fallback = True
            keep_mask[self.group_ids == selected_group] = False
            removed_groups += 1

        kept_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()
        tokens_after = int(kept_positions.numel())
        if tokens_after > self.budget:
            raise RuntimeError("Decoding cache manager failed to enforce its budget.")
        event = CacheTrimEvent(
            logical_length=int(current_logical_length),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_removed=tokens_before - tokens_after,
            groups_removed=removed_groups,
            used_tier2_fallback=used_tier2_fallback,
        )
        return kept_positions, event

    def _apply_trim(
        self, kept_positions: torch.Tensor, event: CacheTrimEvent
    ) -> None:
        self.logical_positions = self.logical_positions[kept_positions]
        self.group_ids = self.group_ids[kept_positions]
        self.token_scores = self.token_scores[kept_positions]
        self.base_tiers = self.base_tiers[kept_positions]
        self.events.append(event)
        if self.initial_post_trim_tokens is None:
            self.initial_post_trim_tokens = event.tokens_after

    def _validate_physical_length(self, physical_length: int) -> None:
        if physical_length != self.cache_length:
            raise ValueError(
                f"Physical cache length {physical_length} does not match "
                f"metadata length {self.cache_length}."
            )


__all__ = ["CacheTrimEvent", "DecodingCacheManager"]
