import pytest
import torch

from src.core.chunker import SentenceBoundaryChunkConstructor
from src.core.evictor import compute_budget_status, compute_keep_mask, evict_kv_cache


def _build_inputs():
    chunks = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([4, 5], dtype=torch.long),
        torch.tensor([6, 7], dtype=torch.long),
        torch.tensor([8, 9], dtype=torch.long),
    ]
    scores = torch.tensor([0.9, 0.1, 0.3, 0.2, 0.8], dtype=torch.float32)
    return chunks, scores


def test_compute_keep_mask_removes_level0_first():
    chunks, scores = _build_inputs()
    tiers = torch.tensor([2, 0, 0, 1, 2], dtype=torch.int8)

    keep_mask, removed = compute_keep_mask(
        mask_tiers=tiers,
        chunk_scores=scores,
        chunks=chunks,
        sequence_length=10,
        budget=6,
    )

    assert removed >= 4
    kept = torch.nonzero(keep_mask, as_tuple=False).flatten().tolist()
    assert kept == [0, 1, 6, 7, 8, 9]


def test_evict_kv_cache_uses_level1_when_level0_insufficient():
    chunks, scores = _build_inputs()
    tiers = torch.tensor([2, 0, 0, 1, 2], dtype=torch.int8)
    k_cache = torch.randn(3, 10, 4)
    v_cache = torch.randn(3, 10, 4)

    result = evict_kv_cache(
        mask_tiers=tiers,
        chunk_scores=scores,
        chunks=chunks,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=4,  # deficit=6 -> remove 4 from level0 + 2 from level1
    )

    kept = result.kept_indices.tolist()
    assert kept == [0, 1, 8, 9]
    assert result.new_k_cache.shape[-2] == len(kept)
    assert result.new_v_cache.shape[-2] == len(kept)


def test_evict_kv_cache_preserves_tier2_when_budget_is_too_strict():
    chunks, scores = _build_inputs()
    tiers = torch.tensor([2, 2, 2, 2, 2], dtype=torch.int8)
    k_cache = torch.randn(2, 10, 8)
    v_cache = torch.randn(2, 10, 8)

    with pytest.warns(UserWarning, match="Unable to satisfy budget"):
        result = evict_kv_cache(
            mask_tiers=tiers,
            chunk_scores=scores,
            chunks=chunks,
            k_cache=k_cache,
            v_cache=v_cache,
            budget=6,
        )

    assert result.kept_indices.tolist() == list(range(10))
    assert result.removed_indices.tolist() == []
    assert result.tokens_removed == 0


def test_bounded_chunks_limit_whole_chunk_budget_underfill():
    max_chunk_tokens = 32
    sequence_length = 200
    budget = 100
    constructor = SentenceBoundaryChunkConstructor(
        tokenizer=None,
        punct_ids={999},
        min_chunk_tokens=1,
        max_chunk_tokens=max_chunk_tokens,
    )
    chunks, _ = constructor.forward(torch.arange(sequence_length))
    scores = torch.arange(len(chunks), dtype=torch.float32)
    tiers = torch.zeros(len(chunks), dtype=torch.int8)

    keep_mask, _ = compute_keep_mask(
        mask_tiers=tiers,
        chunk_scores=scores,
        chunks=chunks,
        sequence_length=sequence_length,
        budget=budget,
    )

    kept_tokens = int(keep_mask.sum().item())
    assert kept_tokens == budget


def test_boundary_refinement_partially_evicts_only_final_ranked_chunk():
    chunks = [
        torch.tensor([3, 0, 2, 1]),
        torch.arange(4, 8),
        torch.arange(8, 10),
    ]
    scores = torch.tensor([0.1, 0.8, 0.9])
    tiers = torch.zeros(3, dtype=torch.int8)
    k_cache = torch.randn(1, 10, 2)
    v_cache = torch.randn(1, 10, 2)

    result = evict_kv_cache(
        mask_tiers=tiers,
        chunk_scores=scores,
        chunks=chunks,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=7,
    )

    assert result.removed_indices.tolist() == [0, 1, 2]
    assert result.kept_indices.tolist() == list(range(3, 10))
    assert result.partially_evicted_chunks == 1
    assert result.budget_shortfall == 0
    assert result.budget_overflow == 0
    assert result.budget_utilization == 1.0


def test_budget_status_uses_sequence_length_when_budget_is_larger():
    status = compute_budget_status(
        sequence_length=8,
        budget=16,
        kept_tokens=8,
    )

    assert status.target_budget == 8
    assert status.shortfall == 0
    assert status.overflow == 0
    assert status.utilization == 1.0
