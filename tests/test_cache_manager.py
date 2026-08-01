import pytest
import torch

from src.models.cache_manager import DecodingCacheManager


def _manager(*, budget=4, recent_window=2):
    chunks = [torch.tensor([i]) for i in range(4)]
    return DecodingCacheManager.from_prompt(
        budget=budget,
        recent_window=recent_window,
        kept_indices=torch.arange(4),
        original_sequence_length=4,
        chunks=chunks,
        chunk_scores=torch.tensor([0.9, 0.1, 0.2, 0.8]),
        mask_tiers=torch.tensor([2, 0, 1, 2], dtype=torch.int8),
    )


def _cache(length):
    values = torch.arange(2 * length, dtype=torch.float32).reshape(1, length, 2)
    return values.clone(), values.clone()


def test_manager_re_evicts_after_every_generated_token():
    manager = _manager()
    k_cache, v_cache = _cache(4)
    k_cache, v_cache, _ = manager.trim_cache_tensors(
        k_cache, v_cache, current_logical_length=4
    )

    for logical_position in range(4, 12):
        manager.append_generated_token(logical_position=logical_position)
        new_entry = torch.full((1, 1, 2), float(logical_position))
        k_cache = torch.cat([k_cache, new_entry], dim=-2)
        v_cache = torch.cat([v_cache, new_entry], dim=-2)
        k_cache, v_cache, event = manager.trim_cache_tensors(
            k_cache,
            v_cache,
            current_logical_length=logical_position + 1,
        )

        assert event.tokens_after <= manager.budget
        assert k_cache.shape[-2] == manager.cache_length
        assert v_cache.shape[-2] == manager.cache_length

    summary = manager.summary()
    assert summary["budget_violations"] == 0
    assert summary["max_post_trim_tokens"] <= manager.budget
    assert summary["re_eviction_events"] > 0
    assert 0 in manager.logical_positions.tolist()
    assert {10, 11}.issubset(set(manager.logical_positions.tolist()))


def test_manager_evicts_lower_tiers_before_sink_and_recent_tokens():
    manager = _manager()
    k_cache, v_cache = _cache(4)
    manager.trim_cache_tensors(k_cache, v_cache, current_logical_length=4)
    manager.append_generated_token(logical_position=4)
    k_cache, v_cache = _cache(5)

    _, _, event = manager.trim_cache_tensors(
        k_cache, v_cache, current_logical_length=5
    )

    retained = set(manager.logical_positions.tolist())
    assert event.used_tier2_fallback is False
    assert 0 in retained
    assert {3, 4}.issubset(retained)
    assert 1 not in retained


def test_manager_uses_tier2_fallback_only_to_guarantee_strict_budget():
    manager = _manager(budget=1, recent_window=4)
    k_cache, v_cache = _cache(4)

    _, _, event = manager.trim_cache_tensors(
        k_cache, v_cache, current_logical_length=4
    )

    assert event.used_tier2_fallback is True
    assert manager.cache_length == 1
    assert manager.logical_positions.tolist() == [0]
    assert manager.summary()["budget_violations"] == 0


def test_manager_compacts_every_dynamic_cache_layer():
    transformers = pytest.importorskip("transformers")
    from transformers.cache_utils import DynamicCache

    manager = _manager()
    dynamic_cache = DynamicCache()
    for layer_idx in range(2):
        key = torch.randn(1, 2, 4, 3)
        value = torch.randn(1, 2, 4, 3)
        dynamic_cache.update(key, value, layer_idx=layer_idx)

    manager.append_generated_token(logical_position=4)
    grown_cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(dynamic_cache.to_legacy_cache()):
        grown_cache.update(
            torch.cat([key, torch.randn(1, 2, 1, 3)], dim=-2),
            torch.cat([value, torch.randn(1, 2, 1, 3)], dim=-2),
            layer_idx=layer_idx,
        )

    compacted, event = manager.trim_past_key_values(
        grown_cache, current_logical_length=5
    )

    assert event.tokens_after <= manager.budget
    for key, value in compacted.to_legacy_cache():
        assert key.shape[-2] == manager.cache_length
        assert value.shape[-2] == manager.cache_length

