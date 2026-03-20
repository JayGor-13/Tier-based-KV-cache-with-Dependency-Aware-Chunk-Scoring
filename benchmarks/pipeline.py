"""Reusable benchmark pipeline utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from benchmarks.eval_metrics import CacheMetrics, compute_cache_metrics
from src.baselines.chunkkv import evict_chunkkv
from src.baselines.h2o import evict_h2o
from src.baselines.snapkv import evict_snapkv
from src.core.evictor import EvictionResult, evict_kv_cache
from src.core.masker import assign_protection_tiers, infer_sequence_length


@dataclass
class TraceSample:
    sample_id: str
    chunks: list[torch.Tensor]
    chunk_scores: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    budget: int | None = None
    attention_obs: torch.Tensor | None = None
    gold: str | None = None
    prediction: str | None = None

    @property
    def sequence_length(self) -> int:
        return int(self.k_cache.shape[-2])


def _to_tensor(data: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype)


def _to_chunks(chunks: Sequence[Sequence[int] | torch.Tensor]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for chunk in chunks:
        if isinstance(chunk, torch.Tensor):
            idx = chunk.to(dtype=torch.long)
        else:
            idx = torch.as_tensor(chunk, dtype=torch.long)
        if idx.ndim != 1:
            raise ValueError("Each chunk must be a 1D list/tensor of token indices.")
        out.append(idx)
    return out


def _make_cache_from_shape(shape: Sequence[int], *, seed: int = 0) -> torch.Tensor:
    if len(shape) < 2:
        raise ValueError(f"Cache shape must have at least 2 dims, got {shape}.")
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(tuple(int(x) for x in shape), generator=g, dtype=torch.float32)


def _fallback_attention_from_chunks(
    chunk_scores: torch.Tensor, chunks: Sequence[torch.Tensor], sequence_length: int
) -> torch.Tensor:
    token_scores = torch.zeros(sequence_length, dtype=torch.float32)
    for chunk_idx, chunk in enumerate(chunks):
        if chunk.numel() == 0:
            continue
        token_scores[chunk] = float(chunk_scores[chunk_idx].item())
    return token_scores.view(1, 1, sequence_length)


def parse_trace_record(record: dict[str, Any], *, sample_index: int = 0) -> TraceSample:
    if "chunks" not in record:
        raise KeyError("Trace record must include `chunks`.")
    if "chunk_scores" not in record:
        raise KeyError("Trace record must include `chunk_scores`.")

    chunks = _to_chunks(record["chunks"])
    chunk_scores = _to_tensor(record["chunk_scores"], dtype=torch.float32)
    if chunk_scores.ndim != 1:
        raise ValueError("`chunk_scores` must be 1D.")
    if chunk_scores.numel() != len(chunks):
        raise ValueError("`chunk_scores` length must match number of chunks.")

    if "k_cache" in record:
        k_cache = _to_tensor(record["k_cache"], dtype=torch.float32)
    else:
        if "k_cache_shape" in record:
            k_shape = record["k_cache_shape"]
        elif "cache_shape" in record:
            k_shape = record["cache_shape"]
        else:
            inferred_t = int(record.get("sequence_length", infer_sequence_length(chunks)))
            num_heads = int(record.get("num_heads", 8))
            head_dim = int(record.get("head_dim", 128))
            k_shape = [num_heads, inferred_t, head_dim]
        k_cache = _make_cache_from_shape(k_shape, seed=sample_index + 17)

    if "v_cache" in record:
        v_cache = _to_tensor(record["v_cache"], dtype=torch.float32)
    else:
        if "v_cache_shape" in record:
            v_shape = record["v_cache_shape"]
        elif "cache_shape" in record:
            v_shape = record["cache_shape"]
        else:
            v_shape = list(k_cache.shape)
        v_cache = _make_cache_from_shape(v_shape, seed=sample_index + 37)

    attention_obs = None
    if "attention_obs" in record:
        attention_obs = _to_tensor(record["attention_obs"], dtype=torch.float32)

    return TraceSample(
        sample_id=str(record.get("id", f"sample_{sample_index}")),
        chunks=chunks,
        chunk_scores=chunk_scores,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=record.get("budget"),
        attention_obs=attention_obs,
        gold=record.get("gold"),
        prediction=record.get("prediction"),
    )


def load_trace_samples(path: str | Path) -> list[TraceSample]:
    path = Path(path)
    suffix = path.suffix.lower()

    raw_records: list[dict[str, Any]]
    if suffix == ".jsonl":
        raw_records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_records.append(json.loads(line))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "samples" in data:
                raw_records = list(data["samples"])
            else:
                raw_records = [data]
        elif isinstance(data, list):
            raw_records = data
        else:
            raise ValueError("JSON trace must be a dict or list.")
    elif suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "samples" in data:
            raw_records = list(data["samples"])
        elif isinstance(data, list):
            raw_records = data
        else:
            raise ValueError("PT trace must contain a list or a dict with `samples`.")
    else:
        raise ValueError(f"Unsupported trace extension `{suffix}`.")

    return [parse_trace_record(rec, sample_index=i) for i, rec in enumerate(raw_records)]


def run_tdc_policy(
    sample: TraceSample,
    *,
    budget: int,
    theta: float = 0.3,
    recent_window: int = 16,
) -> tuple[EvictionResult, torch.Tensor, CacheMetrics]:
    tiers = assign_protection_tiers(
        chunk_scores=sample.chunk_scores,
        chunks=sample.chunks,
        theta=theta,
        recent_window=recent_window,
        sequence_length=sample.sequence_length,
    )
    t0 = time.perf_counter()
    result = evict_kv_cache(
        mask_tiers=tiers,
        chunk_scores=sample.chunk_scores,
        chunks=sample.chunks,
        k_cache=sample.k_cache,
        v_cache=sample.v_cache,
        budget=budget,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    metrics = compute_cache_metrics(
        sample_id=sample.sample_id,
        original_length=sample.sequence_length,
        budget=budget,
        kept_length=int(result.kept_indices.numel()),
        latency_ms=latency_ms,
    )
    return result, tiers, metrics


def run_baseline_policy(
    sample: TraceSample,
    *,
    method: str,
    budget: int,
    recent_window: int = 16,
    theta: float = 0.3,
    heavy_hitter_ratio: float = 0.7,
) -> tuple[EvictionResult, CacheMetrics]:
    method = method.lower()
    if method == "chunkkv":
        result = evict_chunkkv(
            chunk_scores=sample.chunk_scores,
            chunks=sample.chunks,
            k_cache=sample.k_cache,
            v_cache=sample.v_cache,
            budget=budget,
            theta=theta,
            recent_window=recent_window,
            sequence_length=sample.sequence_length,
        )
    else:
        attention_obs = sample.attention_obs
        if attention_obs is None:
            attention_obs = _fallback_attention_from_chunks(
                sample.chunk_scores, sample.chunks, sample.sequence_length
            )
        if method == "snapkv":
            result = evict_snapkv(
                attention_obs=attention_obs,
                k_cache=sample.k_cache,
                v_cache=sample.v_cache,
                budget=budget,
                recent_window=recent_window,
            )
        elif method == "h2o":
            result = evict_h2o(
                attention_obs=attention_obs,
                k_cache=sample.k_cache,
                v_cache=sample.v_cache,
                budget=budget,
                recent_window=recent_window,
                heavy_hitter_ratio=heavy_hitter_ratio,
            )
        else:
            raise ValueError(f"Unsupported baseline method `{method}`.")

    # Latency for baseline operations is not timed here to keep a single metric
    # contract; scripts can wrap with their own timing if needed.
    metrics = compute_cache_metrics(
        sample_id=sample.sample_id,
        original_length=sample.sequence_length,
        budget=budget,
        kept_length=int(result.kept_indices.numel()),
        latency_ms=0.0,
    )
    return result, metrics


__all__ = [
    "TraceSample",
    "load_trace_samples",
    "parse_trace_record",
    "run_baseline_policy",
    "run_tdc_policy",
]
