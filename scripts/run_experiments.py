"""Unified experiment runner for TDC-KV and baselines across benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import compute_cache_metrics, summarize_cache_metrics
from benchmarks.pipeline import (
    load_trace_samples,
    run_baseline_policy,
    run_tdc_full_pipeline_policy,
    run_tdc_policy,
)
from src.core.chunker import MIN_CHUNK_TOKENS


DEFAULT_BENCHMARK_TRACES: dict[str, str] = {
    "niah": "data/niah_trace.jsonl",
    "gsm8k": "data/gsm8k_trace.jsonl",
    "2wiki": "data/2wiki_trace.jsonl",
    "hotpotqa": "data/hotpotqa_trace.jsonl",
    "musique": "data/musique_trace.jsonl",
}

SUPPORTED_METHODS = {"tdc_kv", "chunkkv", "snapkv", "h2o"}


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _normalize_method(method: str) -> str:
    m = method.strip().lower()
    aliases = {
        "our": "tdc_kv",
        "our_model": "tdc_kv",
        "tdc": "tdc_kv",
    }
    return aliases.get(m, m)


def _parse_overrides(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text.strip():
        return out
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                f"Invalid --trace-overrides entry `{piece}`. Use `bench=path`."
            )
        key, value = piece.split("=", 1)
        out[key.strip().lower()] = value.strip()
    return out


def _parse_punct_ids(text: str) -> set[int] | None:
    if text is None:
        return None
    ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    return set(ids) if ids else None


def _resolve_benchmarks(raw: str) -> list[str]:
    selected = [x.lower() for x in _parse_csv(raw)]
    if not selected:
        return list(DEFAULT_BENCHMARK_TRACES.keys())
    if len(selected) == 1 and selected[0] == "all":
        return list(DEFAULT_BENCHMARK_TRACES.keys())
    unsupported = [b for b in selected if b not in DEFAULT_BENCHMARK_TRACES]
    if unsupported:
        raise ValueError(
            f"Unsupported benchmark(s): {unsupported}. "
            f"Allowed: {sorted(DEFAULT_BENCHMARK_TRACES)}"
        )
    return selected


def _resolve_methods(raw: str) -> list[str]:
    methods = [_normalize_method(x) for x in _parse_csv(raw)]
    if not methods:
        methods = ["tdc_kv", "chunkkv", "snapkv", "h2o"]
    unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
    if unsupported:
        raise ValueError(
            f"Unsupported method(s): {unsupported}. Allowed: {sorted(SUPPORTED_METHODS)}"
        )
    return methods


def _aggregate_across_benchmarks(
    per_benchmark: dict[str, dict],
    methods: Iterable[str],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for method in methods:
        all_metrics = []
        for payload in per_benchmark.values():
            bench_method = payload["results"].get(method)
            if not bench_method:
                continue
            all_metrics.extend(bench_method["metric_rows"])
        out[method] = summarize_cache_metrics(all_metrics)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated benchmarks: niah,gsm8k,2wiki,hotpotqa,musique or `all`.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="tdc_kv,chunkkv,snapkv,h2o",
        help="Comma-separated methods: tdc_kv,chunkkv,snapkv,h2o.",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="",
        help="Optional root dir prepended to default benchmark trace paths.",
    )
    parser.add_argument(
        "--trace-overrides",
        type=str,
        default="",
        help="Override traces per benchmark, e.g. `niah=tmp/a.jsonl,gsm8k=tmp/b.jsonl`.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Global budget. If omitted, uses per-sample budget in trace.",
    )
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--recent-window", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--min-chunk-tokens", type=int, default=MIN_CHUNK_TOKENS)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--heavy-hitter-ratio", type=float, default=0.7)
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="For tdc_kv, recompute chunks/scores from token_ids+attention_obs.",
    )
    parser.add_argument(
        "--punct-ids",
        type=str,
        default=None,
        help="Comma-separated punctuation token IDs used by full pipeline mode.",
    )
    parser.add_argument(
        "--allow-level2-fallback",
        action="store_true",
        help="Allow Module 4 to evict Tier-2 chunks when required.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/experiments_results.json",
        help="Combined output JSON.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first sample-level error.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Total runs per sample/method for latency measurement.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of initial runs to discard from latency stats.",
    )
    parser.add_argument(
        "--record-latency-runs",
        action="store_true",
        help="Include per-run latency list in output JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    methods = _resolve_methods(args.methods)
    benchmarks = _resolve_benchmarks(args.benchmarks)
    overrides = _parse_overrides(args.trace_overrides)
    punct_ids = _parse_punct_ids(args.punct_ids)

    trace_root = Path(args.trace_dir) if args.trace_dir else None
    per_benchmark: dict[str, dict] = {}

    for bench in benchmarks:
        default_path = Path(DEFAULT_BENCHMARK_TRACES[bench])
        if bench in overrides:
            trace_path = Path(overrides[bench])
        elif trace_root is not None:
            trace_path = trace_root / default_path.name
        else:
            trace_path = default_path
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Trace file for benchmark `{bench}` not found: {trace_path}"
            )

        samples = load_trace_samples(trace_path)
        bench_result = {"trace_path": str(trace_path), "results": {}}

        for method in methods:
            metric_rows = []
            runs = []
            failed_runs = 0
            for sample in samples:
                budget = int(args.budget if args.budget is not None else (sample.budget or 0))
                if budget <= 0:
                    raise ValueError(
                        f"No valid budget for sample `{sample.sample_id}` in benchmark "
                        f"`{bench}`. Use --budget or include `budget` in traces."
                    )

                try:
                    total_runs = max(1, int(args.repeats))
                    warmup_runs = max(0, min(int(args.warmup_runs), total_runs - 1))
                    latency_runs_ms: list[float] = []

                    result = None
                    tiers = None
                    chunk_scores = None
                    for run_idx in range(total_runs):
                        t0 = time.perf_counter()
                        if method == "tdc_kv":
                            if args.full_pipeline:
                                run_result, run_tiers, _, run_chunk_scores = run_tdc_full_pipeline_policy(
                                    sample,
                                    budget=budget,
                                    theta=args.theta,
                                    recent_window=args.recent_window,
                                    punct_ids=punct_ids,
                                    min_chunk_tokens=args.min_chunk_tokens,
                                    alpha=args.alpha,
                                    beta=args.beta,
                                    window_size=args.window_size,
                                    num_layers=args.num_layers,
                                    allow_level2_fallback=args.allow_level2_fallback,
                                )
                            else:
                                run_result, run_tiers, _ = run_tdc_policy(
                                    sample,
                                    budget=budget,
                                    theta=args.theta,
                                    recent_window=args.recent_window,
                                    allow_level2_fallback=args.allow_level2_fallback,
                                )
                                run_chunk_scores = sample.chunk_scores
                        else:
                            run_result, _ = run_baseline_policy(
                                sample,
                                method=method,
                                budget=budget,
                                recent_window=args.recent_window,
                                theta=args.theta,
                                heavy_hitter_ratio=args.heavy_hitter_ratio,
                            )
                            run_tiers = None
                            run_chunk_scores = None

                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        if run_idx >= warmup_runs:
                            latency_runs_ms.append(elapsed_ms)
                            result = run_result
                            tiers = run_tiers
                            chunk_scores = run_chunk_scores

                    if not latency_runs_ms:
                        raise RuntimeError(
                            "No measured latency runs available after warmup discard."
                        )
                    latency_ms = float(sum(latency_runs_ms) / len(latency_runs_ms))
                except Exception as exc:  # noqa: BLE001
                    if args.fail_fast:
                        raise
                    failed_runs += 1
                    runs.append(
                        {
                            "sample_id": sample.sample_id,
                            "budget": budget,
                            "error": str(exc),
                        }
                    )
                    continue

                metric = compute_cache_metrics(
                    sample_id=sample.sample_id,
                    original_length=sample.sequence_length,
                    budget=budget,
                    kept_length=int(result.kept_indices.numel()),
                    latency_ms=latency_ms,
                )
                metric_rows.append(metric)
                run_row = {
                    "sample_id": sample.sample_id,
                    "budget": budget,
                    "kept_tokens": int(result.kept_indices.numel()),
                    "removed_tokens": int(result.removed_indices.numel()),
                    "repeats": total_runs,
                    "warmup_runs": warmup_runs,
                    "measured_runs": len(latency_runs_ms),
                    "metrics": metric.to_dict(),
                }
                if args.record_latency_runs:
                    run_row["latency_runs_ms"] = latency_runs_ms
                if tiers is not None:
                    run_row.update(
                        {
                            "tier0_chunks": int((tiers == 0).sum().item()),
                            "tier1_chunks": int((tiers == 1).sum().item()),
                            "tier2_chunks": int((tiers == 2).sum().item()),
                        }
                    )
                if chunk_scores is not None:
                    run_row["num_chunks"] = int(chunk_scores.numel())
                runs.append(run_row)

            bench_result["results"][method] = {
                "summary": summarize_cache_metrics(metric_rows),
                "runs": runs,
                "metric_rows": metric_rows,
                "failed_runs": failed_runs,
            }
        per_benchmark[bench] = bench_result

    aggregated = _aggregate_across_benchmarks(per_benchmark, methods)

    cleaned_per_benchmark = {}
    for bench, payload in per_benchmark.items():
        clean_results = {}
        for method, method_payload in payload["results"].items():
            clean_results[method] = {
                "summary": method_payload["summary"],
                "runs": method_payload["runs"],
                "failed_runs": method_payload["failed_runs"],
            }
        cleaned_per_benchmark[bench] = {
            "trace_path": payload["trace_path"],
            "results": clean_results,
        }

    output_payload = {
        "config": {
            "benchmarks": benchmarks,
            "methods": methods,
            "trace_dir": args.trace_dir,
            "trace_overrides": overrides,
            "budget": args.budget,
            "theta": args.theta,
            "recent_window": args.recent_window,
            "window_size": args.window_size,
            "alpha": args.alpha,
            "beta": args.beta,
            "min_chunk_tokens": args.min_chunk_tokens,
            "num_layers": args.num_layers,
            "heavy_hitter_ratio": args.heavy_hitter_ratio,
            "full_pipeline": args.full_pipeline,
            "punct_ids": sorted(punct_ids) if punct_ids is not None else None,
            "allow_level2_fallback": args.allow_level2_fallback,
            "repeats": max(1, int(args.repeats)),
            "warmup_runs": max(0, int(args.warmup_runs)),
            "record_latency_runs": args.record_latency_runs,
        },
        "aggregate_summary": aggregated,
        "benchmarks": cleaned_per_benchmark,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Wrote experiment results to {output_path}")
    print(json.dumps(output_payload["aggregate_summary"], indent=2))


if __name__ == "__main__":
    main()
