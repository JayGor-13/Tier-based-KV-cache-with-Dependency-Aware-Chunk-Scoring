"""Core TDC-KV components."""

from .chunker import (
    MIN_CHUNK_TOKENS,
    SentenceBoundaryChunkConstructor,
    build_module1,
    build_punctuation_vocab,
)
from .evictor import EvictionResult, compute_keep_mask, evict_kv_cache
from .masker import MaskerResult, assign_protection_tiers, find_chunk_index, infer_sequence_length
from .pipeline import TDCKVPipeline, TDCKVPipelineResult, build_tdc_kv_pipeline
from .scorer import DualSignalScorer, build_module2

__all__ = [
    "EvictionResult",
    "MaskerResult",
    "MIN_CHUNK_TOKENS",
    "SentenceBoundaryChunkConstructor",
    "DualSignalScorer",
    "TDCKVPipeline",
    "TDCKVPipelineResult",
    "assign_protection_tiers",
    "build_module1",
    "build_module2",
    "build_punctuation_vocab",
    "build_tdc_kv_pipeline",
    "compute_keep_mask",
    "evict_kv_cache",
    "find_chunk_index",
    "infer_sequence_length",
]
