"""Compatibility helpers for HuggingFace cache layouts across v4 and v5."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch


def _layer_key_value(layer: Any, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(layer, dict):
        key = layer.get("key_states")
        if key is None:
            key = layer.get("key")
        value = layer.get("value_states")
        if value is None:
            value = layer.get("value")
    elif hasattr(layer, "keys") and hasattr(layer, "values"):
        key = layer.keys
        value = layer.values
    elif isinstance(layer, Sequence) and len(layer) >= 2:
        key, value = layer[:2]
    else:
        raise TypeError(
            f"Unsupported cache layer {layer_index} of type "
            f"{type(layer).__name__}."
        )

    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise ValueError(f"Cache layer {layer_index} has no initialized key/value tensors.")
    if key.shape != value.shape:
        raise ValueError(
            f"Cache layer {layer_index} key/value shapes differ: "
            f"{tuple(key.shape)} vs {tuple(value.shape)}."
        )
    return key, value


def cache_layer_tensors(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return cache layers as key/value tensor pairs for HF v4 and v5 caches."""
    if past_key_values is None:
        raise ValueError("Model output did not include `past_key_values`.")

    if hasattr(past_key_values, "layers"):
        raw_layers = list(past_key_values.layers)
    elif hasattr(past_key_values, "to_legacy_cache"):
        raw_layers = list(past_key_values.to_legacy_cache())
    elif hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values,
        "value_cache",
    ):
        raw_layers = list(zip(past_key_values.key_cache, past_key_values.value_cache))
    elif isinstance(past_key_values, Iterable):
        raw_layers = list(past_key_values)
    else:
        raise TypeError(
            "Unsupported past_key_values type: "
            f"{type(past_key_values).__name__}."
        )

    if not raw_layers:
        raise ValueError("past_key_values contains no cache layers.")
    return [
        _layer_key_value(layer, layer_index)
        for layer_index, layer in enumerate(raw_layers)
    ]


def build_dynamic_cache(
    layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> Any:
    """Build a DynamicCache using the update API shared by HF v4 and v5."""
    if not layers:
        raise ValueError("At least one key/value cache layer is required.")

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_index, (key, value) in enumerate(layers):
        if key.shape != value.shape:
            raise ValueError(
                f"Cache layer {layer_index} key/value shapes must match."
            )
        cache.update(key, value, layer_idx=layer_index)
    return cache


__all__ = ["build_dynamic_cache", "cache_layer_tensors"]
