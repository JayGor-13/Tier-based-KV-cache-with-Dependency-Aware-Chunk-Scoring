"""Shared dataset benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.eval_metrics import summarize_cache_metrics, summarize_qa
from benchmarks.pipeline import load_trace_samples, run_tdc_policy


def run_dataset_benchmark(
    *,
    dataset_name: str,
    trace_path: str,
    output_path: str,
    budget: int | None = None,
    theta: float = 0.3,
    recent_window: int = 16,
) -> dict[str, Any]:
    samples = load_trace_samples(trace_path)
    metrics = []
    runs = []
    qa_rows = []

    for sample in samples:
        run_budget = int(budget if budget is not None else (sample.budget or 0))
        if run_budget <= 0:
            raise ValueError(
                f"No valid budget provided for sample `{sample.sample_id}`. "
                "Use --budget or include `budget` in trace records."
            )

        result, tiers, metric = run_tdc_policy(
            sample,
            budget=run_budget,
            theta=theta,
            recent_window=recent_window,
        )
        metrics.append(metric)

        runs.append(
            {
                "sample_id": sample.sample_id,
                "budget": run_budget,
                "kept_tokens": int(result.kept_indices.numel()),
                "removed_tokens": int(result.removed_indices.numel()),
                "tier0_chunks": int((tiers == 0).sum().item()),
                "tier1_chunks": int((tiers == 1).sum().item()),
                "tier2_chunks": int((tiers == 2).sum().item()),
                "metrics": metric.to_dict(),
            }
        )

        if sample.prediction is not None and sample.gold is not None:
            qa_rows.append({"prediction": sample.prediction, "gold": sample.gold})

    payload = {
        "dataset": dataset_name,
        "config": {
            "trace_path": trace_path,
            "budget": budget,
            "theta": theta,
            "recent_window": recent_window,
            "method": "tdc_kv",
        },
        "cache_summary": summarize_cache_metrics(metrics),
        "qa_summary": summarize_qa(qa_rows),
        "runs": runs,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = ["run_dataset_benchmark"]
