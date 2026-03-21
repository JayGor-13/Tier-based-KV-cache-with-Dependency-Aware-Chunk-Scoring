import pytest
import torch

from src.core.evictor import compute_keep_mask, evict_kv_cache


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


def test_evict_kv_cache_raises_when_only_tier2_exists():
    chunks, scores = _build_inputs()
    tiers = torch.tensor([2, 2, 2, 2, 2], dtype=torch.int8)
    k_cache = torch.randn(2, 10, 8)
    v_cache = torch.randn(2, 10, 8)

    with pytest.raises(RuntimeError):
        evict_kv_cache(
            mask_tiers=tiers,
            chunk_scores=scores,
            chunks=chunks,
            k_cache=k_cache,
            v_cache=v_cache,
            budget=6,
        )
