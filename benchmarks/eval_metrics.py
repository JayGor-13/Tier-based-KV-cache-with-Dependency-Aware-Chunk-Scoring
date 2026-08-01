"""Evaluation metrics for cache policies and QA benchmark outputs."""

from __future__ import annotations

from collections import Counter
import json
import re
import statistics
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
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
    compression_multiplier: float
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
    multiplier = (float(original_length) / kept_length) if kept_length > 0 else float("inf")
    gap = kept_length - budget
    return CacheMetrics(
        sample_id=sample_id,
        original_length=original_length,
        budget=budget,
        kept_length=kept_length,
        removed_length=removed,
        retention_ratio=retention,
        compression_ratio=compression,
        compression_multiplier=multiplier,
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
            "avg_compression_multiplier": 0.0,
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

    multipliers = [
        m.compression_multiplier
        for m in metrics
        if isfinite(m.compression_multiplier)
    ]

    return {
        "count": len(metrics),
        "avg_retention_ratio": _mean(m.retention_ratio for m in metrics),
        "avg_compression_ratio": _mean(m.compression_ratio for m in metrics),
        "avg_compression_multiplier": _mean(multipliers) if multipliers else float("inf"),
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


_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_final_answer(text: str) -> str:
    """Extract a final short answer, with GSM8K-style numeric answers in mind."""
    text = str(text or "").strip()
    if not text:
        return ""

    if "####" in text:
        text = text.rsplit("####", 1)[-1]

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text.strip().strip(".:;,$ ")


def _normalize_final_answer(text: str) -> str:
    answer = extract_final_answer(text)
    if not answer:
        return ""

    compact = answer.replace(",", "").replace("$", "").strip()
    try:
        numeric = Decimal(compact)
    except InvalidOperation:
        return _normalize_answer(answer)

    normalized = format(numeric.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def final_answer_exact_match(prediction: str, gold: str) -> float:
    pred = _normalize_final_answer(prediction)
    ref = _normalize_final_answer(gold)
    if not pred or not ref:
        return 0.0
    return float(pred == ref)


def final_answer_f1(prediction: str, gold: str) -> float:
    pred = extract_final_answer(prediction)
    ref = extract_final_answer(gold)
    if not pred or not ref:
        return 0.0
    return token_f1(pred, ref)


def _needs_final_answer_score(rec: dict, gold: str) -> bool:
    dataset = str(rec.get("dataset", "")).lower()
    return "gsm8k" in dataset or "####" in gold


def summarize_qa(predictions: list[dict]) -> dict:
    """Summarize QA metrics for records containing `prediction` and `gold`."""
    if not predictions:
        return {
            "count": 0,
            "exact_match": 0.0,
            "f1": 0.0,
            "final_answer_count": 0,
            "final_answer_exact_match": 0.0,
            "final_answer_f1": 0.0,
        }

    em_scores = []
    f1_scores = []
    final_em_scores = []
    final_f1_scores = []
    for rec in predictions:
        pred = str(rec.get("prediction", ""))
        gold = str(rec.get("gold", ""))
        em_scores.append(exact_match(pred, gold))
        f1_scores.append(token_f1(pred, gold))
        if _needs_final_answer_score(rec, gold):
            final_em_scores.append(final_answer_exact_match(pred, gold))
            final_f1_scores.append(final_answer_f1(pred, gold))

    return {
        "count": len(predictions),
        "exact_match": _mean(em_scores),
        "f1": _mean(f1_scores),
        "final_answer_count": len(final_em_scores),
        "final_answer_exact_match": _mean(final_em_scores),
        "final_answer_f1": _mean(final_f1_scores),
    }


_GROUP_DIMENSION_CONFIG_KEYS = {
    "budget",
    "budget_type",
    "budget_value",
    "budget_specifications",
    "method",
}


def _numeric_field_summary(records: list[dict]) -> dict:
    """Summarize every numeric field present in a collection of dictionaries."""
    numeric_values: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not isfinite(float(value)):
                continue
            numeric_values.setdefault(str(key), []).append(float(value))

    return {
        key: {
            "count": len(values),
            "avg": _mean(values),
            "min": min(values),
            "max": max(values),
            "sum": float(sum(values)),
        }
        for key, values in sorted(numeric_values.items())
    }


def _budget_descriptors(run: dict, config: dict) -> list[dict]:
    specifications = config.get("budget_specifications")
    if isinstance(specifications, list) and specifications:
        return [dict(specification) for specification in specifications]

    budget_type = config.get("budget_type")
    budget_value = config.get("budget_value")
    if budget_type is not None:
        return [{"type": str(budget_type), "value": budget_value}]

    method = str(run.get("method") or config.get("method") or "unknown")
    if method == "fullkv":
        return [{"type": "fullkv", "value": None}]

    resolved_budget = config.get("budget")
    if resolved_budget is None:
        metrics = run.get("metrics") or {}
        resolved_budget = metrics.get("budget")
    if resolved_budget is None:
        return [{"type": "unspecified", "value": None}]
    return [{"type": "resolved", "value": int(resolved_budget)}]


def aggregate_grouped_runs(runs: list[dict]) -> list[dict]:
    """Aggregate raw runs by model, dataset, method, budget, and configuration.

    Ratio-budget groups use their requested ratio rather than the per-sample
    resolved token count. This keeps variable-length samples in the same paper
    result group while retaining resolved-budget statistics in the summary.
    """
    grouped: dict[str, dict] = {}

    for run in runs:
        config = dict(run.get("config") or {})
        method = str(run.get("method") or config.get("method") or "unknown")
        configuration = {
            key: config[key]
            for key in sorted(config)
            if key not in _GROUP_DIMENSION_CONFIG_KEYS
        }
        for budget in _budget_descriptors(run, config):
            dimensions = {
                "model": str(run.get("model") or "unknown"),
                "dataset": str(run.get("dataset") or "unknown"),
                "method": method,
                "budget": budget,
                "configuration": configuration,
            }
            group_key = json.dumps(dimensions, sort_keys=True, separators=(",", ":"))
            bucket = grouped.setdefault(
                group_key,
                {
                    "dimensions": dimensions,
                    "runs": [],
                },
            )
            bucket["runs"].append(run)

    results: list[dict] = []
    cache_fields = set(CacheMetrics.__dataclass_fields__)
    for group_key in sorted(grouped):
        bucket = grouped[group_key]
        group_runs = bucket["runs"]
        successful = [run for run in group_runs if run.get("status") == "ok"]
        failed = [run for run in group_runs if run.get("status") != "ok"]

        cache_metrics: list[CacheMetrics] = []
        qa_rows: list[dict] = []
        sequence_records: list[dict] = []
        resolved_budget_records: list[dict] = []
        decode_records: list[dict] = []
        tier_records: list[dict] = []

        for run in successful:
            metric_payload = run.get("metrics")
            if isinstance(metric_payload, dict) and cache_fields.issubset(metric_payload):
                cache_metrics.append(
                    CacheMetrics(
                        **{key: metric_payload[key] for key in cache_fields}
                    )
                )

            prediction = run.get("evicted_prediction", run.get("prediction"))
            gold = run.get("gold")
            if prediction not in {None, ""} and gold is not None:
                qa_rows.append(
                    {
                        "prediction": str(prediction),
                        "gold": str(gold),
                        "dataset": str(run.get("dataset") or ""),
                    }
                )

            if isinstance(run.get("sequence_length"), (int, float)):
                sequence_records.append(
                    {"sequence_length": run["sequence_length"]}
                )
            resolved_budget = (run.get("config") or {}).get("budget")
            if resolved_budget is None and isinstance(metric_payload, dict):
                resolved_budget = metric_payload.get("budget")
            budget_record = {}
            if isinstance(resolved_budget, (int, float)):
                budget_record["resolved_budget"] = resolved_budget
            if isinstance(run.get("kept_tokens"), (int, float)):
                budget_record["kept_tokens"] = run["kept_tokens"]
            if budget_record:
                resolved_budget_records.append(budget_record)
            if isinstance(run.get("decode_cache_summary"), dict):
                decode_records.append(run["decode_cache_summary"])
            if isinstance(run.get("tier_counts"), dict):
                tier_records.append(run["tier_counts"])

        error_types = Counter(str(run.get("error_type") or "UnknownError") for run in failed)
        dimensions = bucket["dimensions"]
        results.append(
            {
                "model": dimensions["model"],
                "dataset": dimensions["dataset"],
                "method": dimensions["method"],
                "budget": dimensions["budget"],
                "configuration": dimensions["configuration"],
                "run_summary": {
                    "total": len(group_runs),
                    "successful": len(successful),
                    "failed": len(failed),
                    "unique_samples": len(
                        {str(run.get("sample_id")) for run in group_runs}
                    ),
                    "error_types": dict(sorted(error_types.items())),
                },
                "cache_summary": summarize_cache_metrics(cache_metrics),
                "qa_summary": summarize_qa(qa_rows),
                "sequence_summary": _numeric_field_summary(sequence_records),
                "resolved_budget_summary": _numeric_field_summary(
                    resolved_budget_records
                ),
                "decode_cache_summary": _numeric_field_summary(decode_records),
                "tier_summary": _numeric_field_summary(tier_records),
            }
        )

    return results


__all__ = [
    "CacheMetrics",
    "aggregate_grouped_runs",
    "compute_cache_metrics",
    "exact_match",
    "extract_final_answer",
    "final_answer_exact_match",
    "final_answer_f1",
    "summarize_cache_metrics",
    "summarize_qa",
    "token_f1",
]
