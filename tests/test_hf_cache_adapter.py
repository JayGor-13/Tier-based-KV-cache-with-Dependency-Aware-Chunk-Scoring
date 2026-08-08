from types import SimpleNamespace

import pytest
import torch

from src.models.hf_cache_adapter import build_dynamic_cache, cache_layer_tensors


def _layers(count=2, length=4):
    return [
        (
            torch.randn(1, 2, length, 3),
            torch.randn(1, 2, length, 3),
        )
        for _ in range(count)
    ]


def test_extracts_transformers_v5_non_subscriptable_layers():
    expected = _layers()
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(keys=key, values=value)
            for key, value in expected
        ]
    )

    actual = cache_layer_tensors(cache)

    assert len(actual) == len(expected)
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        actual,
        expected,
    ):
        assert actual_key is expected_key
        assert actual_value is expected_value


def test_extracts_legacy_tuple_cache():
    expected = _layers()

    actual = cache_layer_tensors(tuple(expected))

    assert len(actual) == len(expected)
    assert all(torch.equal(a[0], e[0]) for a, e in zip(actual, expected))
    assert all(torch.equal(a[1], e[1]) for a, e in zip(actual, expected))


def test_rejects_uninitialized_v5_layer():
    cache = SimpleNamespace(layers=[SimpleNamespace()])

    with pytest.raises(TypeError, match="Unsupported cache layer 0"):
        cache_layer_tensors(cache)


def test_dynamic_cache_round_trip_uses_shared_update_api():
    pytest.importorskip("transformers")
    expected = _layers(count=2, length=5)

    cache = build_dynamic_cache(expected)
    actual = cache_layer_tensors(cache)

    assert len(actual) == 2
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        actual,
        expected,
    ):
        assert torch.equal(actual_key, expected_key)
        assert torch.equal(actual_value, expected_value)
