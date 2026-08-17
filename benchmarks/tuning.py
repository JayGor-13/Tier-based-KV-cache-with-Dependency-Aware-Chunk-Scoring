"""Select a frozen TDC-KV configuration from the tuning experiment."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from benchmarks.eval_metrics import (
    gsm8k_accuracy,
    hotpotqa_f1,
    niah_retrieval_match,
    token_f1,
)


def _score(run: dict[str, Any]) -> float | None:
    gold = run.get("gold")
    if gold is None:
        return None
    prediction = str(run.get("evicted_prediction", ""))
    gold = str(gold)
    dataset = str(run.get("dataset", "")).lower()
    if "gsm8k" in dataset:
        return gsm8k_accuracy(prediction, gold)
    if "niah" in dataset or "needle" in dataset:
        return niah_retrieval_match(prediction, gold)
    if "hotpot" in dataset:
        return hotpotqa_f1(prediction, gold)
    return token_f1(prediction, gold)


def select_best_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    """Choose the best macro-dataset score, breaking ties by policy overhead."""
    grouped: dict[str, dict[str, Any]] = {}
    for run in payload.get("runs", []):
        if run.get("status") != "ok" or run.get("method") != "tdc_kv":
            continue
        config = run.get("config") or {}
        candidate = {
            "alpha": float(config.get("alpha", 0.6)),
            "theta": float(config.get("theta", 0.3)),
            "recent_window": int(config.get("recent_window", 16)),
            "tier1_score_mode": str(config.get("tier1_score_mode", "dependency")),
            "chunking_strategy": str(config.get("chunking_strategy", "sentence")),
            "fixed_chunk_size": int(config.get("fixed_chunk_size", 16)),
            "layer_weighting": str(config.get("layer_weighting", "linear")),
            "protect_sink": bool(config.get("protect_sink", True)),
            "protect_recent": bool(config.get("protect_recent", True)),
        }
        key = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        bucket = grouped.setdefault(
            key,
            {
                "configuration": candidate,
                "dataset_scores": defaultdict(list),
                "policy_ms": [],
            },
        )
        score = _score(run)
        if score is not None:
            bucket["dataset_scores"][str(run.get("dataset"))].append(float(score))
        policy_ms = (
            (run.get("runtime") or {}).get("stages", {}).get("policy", {}).get("elapsed_ms")
        )
        if policy_ms is not None:
            bucket["policy_ms"].append(float(policy_ms))

    candidates = []
    for bucket in grouped.values():
        dataset_means = {
            dataset: sum(values) / len(values)
            for dataset, values in bucket["dataset_scores"].items()
            if values
        }
        if not dataset_means:
            continue
        candidates.append(
            {
                "configuration": bucket["configuration"],
                "macro_dataset_score": sum(dataset_means.values()) / len(dataset_means),
                "dataset_scores": dataset_means,
                "avg_policy_ms": (
                    sum(bucket["policy_ms"]) / len(bucket["policy_ms"])
                    if bucket["policy_ms"]
                    else None
                ),
            }
        )
    if not candidates:
        raise ValueError("The tuning result contains no successful scored TDC-KV runs.")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["macro_dataset_score"],
            row["avg_policy_ms"] if row["avg_policy_ms"] is not None else float("inf"),
            json.dumps(row["configuration"], sort_keys=True),
        ),
    )
    return {
        "selection_rule": "highest_macro_dataset_score_then_lowest_policy_ms",
        "selected": ranked[0],
        "candidate_count": len(ranked),
        "ranking": ranked,
    }


def select_from_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    selection = select_best_configuration(payload)
    selection["source"] = str(input_path.resolve())
    selection["experiment_fingerprint"] = payload.get("experiment_fingerprint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


__all__ = ["select_best_configuration", "select_from_file"]
