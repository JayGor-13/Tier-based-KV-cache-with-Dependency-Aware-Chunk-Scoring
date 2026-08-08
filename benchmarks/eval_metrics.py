"""Evaluation metrics for cache policies and QA benchmark outputs."""

from __future__ import annotations

from collections import Counter
import json
import re
import statistics
import string
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Iterable

from src.core.evictor import compute_budget_status


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
    target_budget: int
    budget_shortfall: int
    budget_overflow: int
    budget_utilization: float
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
    budget_status = compute_budget_status(
        sequence_length=original_length,
        budget=budget,
        kept_tokens=kept_length,
    )
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
        target_budget=budget_status.target_budget,
        budget_shortfall=budget_status.shortfall,
        budget_overflow=budget_status.overflow,
        budget_utilization=budget_status.utilization,
        latency_ms=float(latency_ms),
    )


def validate_budget_contract(
    metrics: CacheMetrics,
    *,
    min_utilization: float = 0.99,
    max_shortfall_tokens: int = 1,
    allow_overflow: bool = False,
) -> None:
    """Require a matched-budget result within explicit occupancy tolerances."""
    if not 0.0 <= min_utilization <= 1.0:
        raise ValueError("min_utilization must be in [0, 1].")
    if max_shortfall_tokens < 0:
        raise ValueError("max_shortfall_tokens must be non-negative.")
    if metrics.budget_overflow > 0 and not allow_overflow:
        raise ValueError(
            f"Cache exceeds target budget by {metrics.budget_overflow} tokens."
        )
    if metrics.budget_shortfall > max_shortfall_tokens:
        raise ValueError(
            "Cache budget shortfall "
            f"{metrics.budget_shortfall} exceeds tolerance "
            f"{max_shortfall_tokens}."
        )
    if metrics.budget_utilization < min_utilization:
        raise ValueError(
            "Cache budget utilization "
            f"{metrics.budget_utilization:.6f} is below required "
            f"{min_utilization:.6f}."
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
            "avg_budget_utilization": 0.0,
            "min_budget_utilization": 0.0,
            "avg_budget_shortfall": 0.0,
            "max_budget_shortfall": 0,
            "budget_overflow_count": 0,
            "exact_budget_match_rate": 0.0,
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
        "avg_budget_utilization": _mean(m.budget_utilization for m in metrics),
        "min_budget_utilization": min(m.budget_utilization for m in metrics),
        "avg_budget_shortfall": _mean(float(m.budget_shortfall) for m in metrics),
        "max_budget_shortfall": max(m.budget_shortfall for m in metrics),
        "budget_overflow_count": sum(1 for m in metrics if m.budget_overflow > 0),
        "exact_budget_match_rate": _mean(
            1.0
            if m.budget_shortfall == 0 and m.budget_overflow == 0
            else 0.0
            for m in metrics
        ),
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

    numeric_text = text.replace("-$", "-").replace("$-", "-").replace("$", "")
    numbers = _NUMBER_RE.findall(numeric_text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text.strip().strip(".:;,$ ")


def normalize_final_answer(text: str) -> str:
    """Normalize a final numerical answer for auditable GSM8K judging."""
    answer = extract_final_answer(text)
    if not answer:
        return ""

    compact = answer.replace(",", "").replace("$", "").strip()
    try:
        numeric = Decimal(compact)
    except InvalidOperation:
        return _normalize_answer(answer)

    if numeric == 0:
        return "0"
    normalized = format(numeric.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def final_answer_exact_match(prediction: str, gold: str) -> float:
    pred = normalize_final_answer(prediction)
    ref = normalize_final_answer(gold)
    if not pred or not ref:
        return 0.0
    return float(pred == ref)


def final_answer_f1(prediction: str, gold: str) -> float:
    pred = extract_final_answer(prediction)
    ref = extract_final_answer(gold)
    if not pred or not ref:
        return 0.0
    return token_f1(pred, ref)


def gsm8k_accuracy(prediction: str, gold: str) -> float:
    """Score a GSM8K completion by its extracted final numeric answer."""
    return final_answer_exact_match(prediction, gold)


def judge_gsm8k_prediction(
    prediction: str,
    gold: str,
    *,
    protocol: str,
) -> dict:
    """Return the complete per-sample record for GSM8K numerical judging."""
    normalized_prediction = normalize_final_answer(prediction)
    normalized_gold = normalize_final_answer(gold)
    score = float(
        bool(normalized_prediction)
        and bool(normalized_gold)
        and normalized_prediction == normalized_gold
    )
    return {
        "judge": "gsm8k_final_numeric_exact_match",
        "judge_version": 1,
        "protocol": str(protocol),
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "score": score,
        "correct": bool(score),
    }


def summarize_generation_parity(records: list[dict]) -> dict:
    """Summarize FullKV parity controls from per-sample comparison records."""
    if not records:
        return {
            "count": 0,
            "text_matches": 0,
            "token_matches": 0,
            "text_parity_rate": None,
            "token_parity_rate": None,
            "all_passed": None,
            "mismatched_samples": [],
        }

    text_matches = sum(bool(record.get("text_match")) for record in records)
    token_matches = sum(bool(record.get("token_match")) for record in records)
    mismatched_samples = [
        str(record.get("sample_id"))
        for record in records
        if not record.get("text_match") or not record.get("token_match")
    ]
    count = len(records)
    return {
        "count": count,
        "text_matches": text_matches,
        "token_matches": token_matches,
        "text_parity_rate": text_matches / float(count),
        "token_parity_rate": token_matches / float(count),
        "all_passed": text_matches == count and token_matches == count,
        "mismatched_samples": mismatched_samples,
    }


def niah_retrieval_match(prediction: str, gold: str) -> float:
    """Return one when the normalized needle occurs as a full token sequence."""
    prediction_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return 0.0

    width = len(gold_tokens)
    return float(
        any(
            prediction_tokens[start : start + width] == gold_tokens
            for start in range(len(prediction_tokens) - width + 1)
        )
    )


def _normalize_hotpotqa_answer(text: str) -> str:
    """Apply the normalization used by the official HotpotQA evaluator."""
    lowered = str(text or "").lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def hotpotqa_exact_match(prediction: str, gold: str) -> float:
    """Compute official normalized answer exact match for HotpotQA."""
    return float(
        _normalize_hotpotqa_answer(prediction) == _normalize_hotpotqa_answer(gold)
    )


def _hotpotqa_prf(prediction: str, gold: str) -> tuple[float, float, float]:
    normalized_prediction = _normalize_hotpotqa_answer(prediction)
    normalized_gold = _normalize_hotpotqa_answer(gold)
    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers
        or normalized_gold in special_answers
    ) and normalized_prediction != normalized_gold:
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1, precision, recall


def hotpotqa_f1(prediction: str, gold: str) -> float:
    """Compute the answer F1 used by the official HotpotQA evaluator."""
    f1, _, _ = _hotpotqa_prf(prediction, gold)
    return f1


def _dataset_family(dataset: str) -> str:
    normalized = str(dataset or "").strip().lower().replace("-", "_")
    if "gsm8k" in normalized:
        return "gsm8k"
    if "niah" in normalized or "needle" in normalized:
        return "niah"
    if "hotpot" in normalized:
        return "hotpotqa"
    return "generic"


def _needs_final_answer_score(rec: dict, gold: str) -> bool:
    return _dataset_family(str(rec.get("dataset", ""))) == "gsm8k" or "####" in gold


def _summarize_task_family(family: str, records: list[dict]) -> dict:
    pairs = [
        (str(record.get("prediction", "")), str(record.get("gold", "")))
        for record in records
    ]

    if family == "gsm8k":
        accuracy = _mean(gsm8k_accuracy(prediction, gold) for prediction, gold in pairs)
        answer_f1 = _mean(final_answer_f1(prediction, gold) for prediction, gold in pairs)
        return {
            "count": len(records),
            "primary_metric": "gsm8k_accuracy",
            "primary_score": accuracy,
            "secondary_metric": "gsm8k_final_answer_f1",
            "secondary_score": answer_f1,
            "accuracy": accuracy,
            "final_answer_f1": answer_f1,
        }

    if family == "niah":
        retrieval_accuracy = _mean(
            niah_retrieval_match(prediction, gold) for prediction, gold in pairs
        )
        normalized_em = _mean(exact_match(prediction, gold) for prediction, gold in pairs)
        return {
            "count": len(records),
            "primary_metric": "niah_retrieval_accuracy",
            "primary_score": retrieval_accuracy,
            "secondary_metric": "niah_exact_match",
            "secondary_score": normalized_em,
            "retrieval_accuracy": retrieval_accuracy,
            "exact_match": normalized_em,
        }

    if family == "hotpotqa":
        prf_scores = [
            _hotpotqa_prf(prediction, gold) for prediction, gold in pairs
        ]
        answer_f1 = _mean(score[0] for score in prf_scores)
        answer_precision = _mean(score[1] for score in prf_scores)
        answer_recall = _mean(score[2] for score in prf_scores)
        answer_em = _mean(
            hotpotqa_exact_match(prediction, gold) for prediction, gold in pairs
        )
        return {
            "count": len(records),
            "primary_metric": "hotpotqa_f1",
            "primary_score": answer_f1,
            "secondary_metric": "hotpotqa_exact_match",
            "secondary_score": answer_em,
            "f1": answer_f1,
            "precision": answer_precision,
            "recall": answer_recall,
            "exact_match": answer_em,
        }

    generic_f1 = _mean(token_f1(prediction, gold) for prediction, gold in pairs)
    generic_em = _mean(exact_match(prediction, gold) for prediction, gold in pairs)
    return {
        "count": len(records),
        "primary_metric": "token_f1",
        "primary_score": generic_f1,
        "secondary_metric": "exact_match",
        "secondary_score": generic_em,
        "f1": generic_f1,
        "exact_match": generic_em,
    }


def summarize_qa(predictions: list[dict]) -> dict:
    """Summarize generic diagnostics and dataset-specific primary metrics."""
    if not predictions:
        return {
            "count": 0,
            "exact_match": 0.0,
            "f1": 0.0,
            "final_answer_count": 0,
            "final_answer_exact_match": 0.0,
            "final_answer_f1": 0.0,
            "task_family": None,
            "primary_metric": None,
            "primary_score": None,
            "secondary_metric": None,
            "secondary_score": None,
            "dataset_metrics": {},
        }

    em_scores = []
    f1_scores = []
    final_em_scores = []
    final_f1_scores = []
    family_records: dict[str, list[dict]] = {}
    for rec in predictions:
        pred = str(rec.get("prediction", ""))
        gold = str(rec.get("gold", ""))
        em_scores.append(exact_match(pred, gold))
        f1_scores.append(token_f1(pred, gold))
        family = _dataset_family(str(rec.get("dataset", "")))
        family_records.setdefault(family, []).append(rec)
        if _needs_final_answer_score(rec, gold):
            final_em_scores.append(final_answer_exact_match(pred, gold))
            final_f1_scores.append(final_answer_f1(pred, gold))

    dataset_metrics = {
        family: _summarize_task_family(family, records)
        for family, records in sorted(family_records.items())
    }
    if len(dataset_metrics) == 1:
        task_family, task_summary = next(iter(dataset_metrics.items()))
        primary_metric = task_summary["primary_metric"]
        primary_score = task_summary["primary_score"]
        secondary_metric = task_summary["secondary_metric"]
        secondary_score = task_summary["secondary_score"]
    else:
        task_family = "mixed"
        primary_metric = "macro_task_score"
        primary_score = _mean(
            summary["primary_score"] for summary in dataset_metrics.values()
        )
        secondary_metric = None
        secondary_score = None

    return {
        "count": len(predictions),
        "exact_match": _mean(em_scores),
        "f1": _mean(f1_scores),
        "final_answer_count": len(final_em_scores),
        "final_answer_exact_match": _mean(final_em_scores),
        "final_answer_f1": _mean(final_f1_scores),
        "task_family": task_family,
        "primary_metric": primary_metric,
        "primary_score": primary_score,
        "secondary_metric": secondary_metric,
        "secondary_score": secondary_score,
        "dataset_metrics": dataset_metrics,
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
    required_cache_fields = {
        "sample_id",
        "original_length",
        "budget",
        "kept_length",
        "latency_ms",
    }
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
        chunk_records: list[dict] = []

        for run in successful:
            metric_payload = run.get("metrics")
            if (
                isinstance(metric_payload, dict)
                and required_cache_fields.issubset(metric_payload)
            ):
                cache_metrics.append(
                    compute_cache_metrics(
                        sample_id=str(metric_payload["sample_id"]),
                        original_length=int(metric_payload["original_length"]),
                        budget=int(metric_payload["budget"]),
                        kept_length=int(metric_payload["kept_length"]),
                        latency_ms=float(metric_payload["latency_ms"]),
                    )
                )

            prediction = run.get("evicted_prediction", run.get("prediction"))
            gold = run.get("gold")
            if prediction is not None and gold is not None:
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
            chunk_record = {
                field: run[field]
                for field in (
                    "num_chunks",
                    "min_chunk_size",
                    "max_chunk_size",
                    "avg_chunk_size",
                    "partially_evicted_chunks",
                )
                if isinstance(run.get(field), (int, float))
            }
            if chunk_record:
                chunk_records.append(chunk_record)

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
                "chunk_summary": _numeric_field_summary(chunk_records),
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
    "gsm8k_accuracy",
    "hotpotqa_exact_match",
    "hotpotqa_f1",
    "judge_gsm8k_prediction",
    "niah_retrieval_match",
    "normalize_final_answer",
    "summarize_generation_parity",
    "summarize_cache_metrics",
    "summarize_qa",
    "token_f1",
    "validate_budget_contract",
]
