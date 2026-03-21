"""Evaluation metrics for cache policies and QA benchmark outputs."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class CacheMetrics:
    sample_id: str
    original_length: int
    budget: int
    kept_length: int
    removed_length: int
    retention_ratio: float
    compression_ratio: float
    budget_gap: int
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_cache_metrics(
    *,
    sample_id: str,
    original_length: int,
    budget: int,
    kept_length: int,
    latency_ms: float,
) -> CacheMetrics:
    removed = max(original_length - kept_length, 0)
    retention = (kept_length / float(original_length)) if original_length > 0 else 1.0
    compression = 1.0 - retention
    gap = kept_length - budget
    return CacheMetrics(
        sample_id=sample_id,
        original_length=original_length,
        budget=budget,
        kept_length=kept_length,
        removed_length=removed,
        retention_ratio=retention,
        compression_ratio=compression,
        budget_gap=gap,
        latency_ms=float(latency_ms),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def summarize_cache_metrics(metrics: list[CacheMetrics]) -> dict:
    """Aggregate cache metrics with average and percentile latency."""
    if not metrics:
        return {
            "count": 0,
            "avg_retention_ratio": 0.0,
            "avg_compression_ratio": 0.0,
            "avg_budget_gap": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p90_latency_ms": 0.0,
        }

    latencies = [m.latency_ms for m in metrics]
    sorted_lat = sorted(latencies)
    p50 = statistics.median(sorted_lat)
    p90_idx = int(0.9 * (len(sorted_lat) - 1))
    p90 = sorted_lat[p90_idx]

    return {
        "count": len(metrics),
        "avg_retention_ratio": _mean(m.retention_ratio for m in metrics),
        "avg_compression_ratio": _mean(m.compression_ratio for m in metrics),
        "avg_budget_gap": _mean(float(m.budget_gap) for m in metrics),
        "avg_latency_ms": _mean(latencies),
        "p50_latency_ms": float(p50),
        "p90_latency_ms": float(p90),
    }


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = _normalize_answer(prediction).split()
    gold_toks = _normalize_answer(gold).split()

    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    common = {}
    for tok in pred_toks:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_toks:
        cnt = common.get(tok, 0)
        if cnt > 0:
            overlap += 1
            common[tok] = cnt - 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2.0 * precision * recall / (precision + recall)


def summarize_qa(predictions: list[dict]) -> dict:
    """Summarize QA metrics for records containing `prediction` and `gold`."""
    if not predictions:
        return {"count": 0, "exact_match": 0.0, "f1": 0.0}

    em_scores = []
    f1_scores = []
    for rec in predictions:
        pred = str(rec.get("prediction", ""))
        gold = str(rec.get("gold", ""))
        em_scores.append(exact_match(pred, gold))
        f1_scores.append(token_f1(pred, gold))

    return {
        "count": len(predictions),
        "exact_match": _mean(em_scores),
        "f1": _mean(f1_scores),
    }


__all__ = [
    "CacheMetrics",
    "compute_cache_metrics",
    "exact_match",
    "summarize_cache_metrics",
    "summarize_qa",
    "token_f1",
]
