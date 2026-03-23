"""Model integration helpers for TDC-KV."""

from src.models.cache_utils import (
    PastKeyValueEvictionResult,
    apply_keep_indices_to_past_key_values,
    cache_sequence_length,
    compute_eviction_indices,
    evict_past_key_values,
    first_layer_cache_tensors,
    last_layer_observed_attention,
    stack_observed_attention,
)
from src.models.modeling_llama import CacheCompressionOutput, LlamaTDCKVModel
from src.models.modeling_phi3 import Phi3TDCKVModel

__all__ = [
    "CacheCompressionOutput",
    "LlamaTDCKVModel",
    "PastKeyValueEvictionResult",
    "Phi3TDCKVModel",
    "apply_keep_indices_to_past_key_values",
    "cache_sequence_length",
    "compute_eviction_indices",
    "evict_past_key_values",
    "first_layer_cache_tensors",
    "last_layer_observed_attention",
    "stack_observed_attention",
]
