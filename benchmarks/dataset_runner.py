"""Shared dataset benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.eval_metrics import summarize_cache_metrics, summarize_qa
from benchmarks.pipeline import (
    load_trace_samples,
    run_tdc_full_pipeline_policy,
    run_tdc_policy,
)
from src.core.chunker import MIN_CHUNK_TOKENS


def run_dataset_benchmark(
    *,
    dataset_name: str,
    trace_path: str,
    output_path: str,
    budget: int | None = None,
    theta: float = 0.3,
    recent_window: int = 16,
    method: str = "tdc_kv",
    full_pipeline: bool = False,
    punct_ids: set[int] | None = None,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    alpha: float = 0.6,
    beta: float = 0.4,
    window_size: int = 16,
    num_layers: int | None = None,
    allow_level2_fallback: bool = False,
) -> dict[str, Any]:
    samples = load_trace_samples(trace_path)
    metrics = []
    runs = []
    qa_rows = []
    method = method.lower()

    for sample in samples:
        run_budget = int(budget if budget is not None else (sample.budget or 0))
        if run_budget <= 0:
            raise ValueError(
                f"No valid budget provided for sample `{sample.sample_id}`. "
                "Use --budget or include `budget` in trace records."
            )

        if method == "tdc_kv":
            if full_pipeline:
                result, tiers, metric, chunk_scores = run_tdc_full_pipeline_policy(
                    sample,
                    budget=run_budget,
                    theta=theta,
                    recent_window=recent_window,
                    punct_ids=punct_ids,
                    min_chunk_tokens=min_chunk_tokens,
                    alpha=alpha,
                    beta=beta,
                    window_size=window_size,
                    num_layers=num_layers,
                    allow_level2_fallback=allow_level2_fallback,
                )
            else:
                result, tiers, metric = run_tdc_policy(
                    sample,
                    budget=run_budget,
                    theta=theta,
                    recent_window=recent_window,
                    allow_level2_fallback=allow_level2_fallback,
                )
                chunk_scores = sample.chunk_scores
        else:
            raise ValueError(
                f"Unsupported method `{method}` for dataset runner. "
                "Use `tdc_kv`."
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
                "num_chunks": int(chunk_scores.numel()),
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
            "method": method,
            "full_pipeline": full_pipeline,
            "min_chunk_tokens": min_chunk_tokens,
            "alpha": alpha,
            "beta": beta,
            "window_size": window_size,
            "num_layers": num_layers,
            "allow_level2_fallback": allow_level2_fallback,
            "punct_ids": sorted(punct_ids) if punct_ids is not None else None,
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
