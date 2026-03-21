"""Core TDC-KV components."""

from .evictor import EvictionResult, compute_keep_mask, evict_kv_cache
from .masker import MaskerResult, assign_protection_tiers, find_chunk_index, infer_sequence_length
from .chunker import (
    DEFAULT_BOUNDARY_CHARS,
    MIN_CHUNK_TOKENS,
    SentenceBoundaryChunkConstructor,
    build_module1,
    build_punctuation_vocab,
    chunk_token_ids,
)
from .scorer import (
    DualSignalScorer,
    build_module2
)
__all__ = [
    "DEFAULT_BOUNDARY_CHARS",
    "MIN_CHUNK_TOKENS",
    "SentenceBoundaryChunkConstructor",
    "build_module1",
    "build_punctuation_vocab",
    "chunk_token_ids",
    "DualSignalScorer",
    "build_module2",
    "EvictionResult",
    "MaskerResult",
    "assign_protection_tiers",
    "compute_keep_mask",
    "evict_kv_cache",
    "find_chunk_index",
    "infer_sequence_length",
]
