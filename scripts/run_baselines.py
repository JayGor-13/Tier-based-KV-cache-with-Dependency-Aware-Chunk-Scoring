"""Run baseline comparisons (ChunkKV, SnapKV, H2O) on trace inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import compute_cache_metrics, summarize_cache_metrics
from benchmarks.pipeline import load_trace_samples, run_baseline_policy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=str, required=True, help="JSON/JSONL/PT trace input.")
    parser.add_argument(
        "--methods",
        type=str,
        default="chunkkv,snapkv,h2o",
        help="Comma-separated baseline methods.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Global budget. If omitted, uses per-sample budget in trace.",
    )
    parser.add_argument("--theta", type=float, default=0.3, help="ChunkKV keep ratio.")
    parser.add_argument("--recent-window", type=int, default=16, help="Recent token keep window.")
    parser.add_argument(
        "--heavy-hitter-ratio",
        type=float,
        default=0.7,
        help="H2O heavy-hitter ratio inside remaining capacity.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/baselines_results.json",
        help="Output JSON file.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    samples = load_trace_samples(args.trace_path)

    all_results: dict[str, dict] = {}
    for method in methods:
        method_runs = []
        method_metrics = []
        for sample in samples:
            budget = int(args.budget if args.budget is not None else (sample.budget or 0))
            if budget <= 0:
                raise ValueError(
                    f"No valid budget provided for sample `{sample.sample_id}`. "
                    "Use --budget or include `budget` in trace records."
                )

            t0 = time.perf_counter()
            result, _ = run_baseline_policy(
                sample,
                method=method,
                budget=budget,
                recent_window=args.recent_window,
                theta=args.theta,
                heavy_hitter_ratio=args.heavy_hitter_ratio,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics = compute_cache_metrics(
                sample_id=sample.sample_id,
                original_length=sample.sequence_length,
                budget=budget,
                kept_length=int(result.kept_indices.numel()),
                latency_ms=latency_ms,
            )
            method_metrics.append(metrics)
            method_runs.append(
                {
                    "sample_id": sample.sample_id,
                    "budget": budget,
                    "kept_tokens": int(result.kept_indices.numel()),
                    "removed_tokens": int(result.removed_indices.numel()),
                    "metrics": metrics.to_dict(),
                }
            )

        all_results[method] = {
            "summary": summarize_cache_metrics(method_metrics),
            "runs": method_runs,
        }

    payload = {
        "config": {
            "trace_path": args.trace_path,
            "methods": methods,
            "budget": args.budget,
            "theta": args.theta,
            "recent_window": args.recent_window,
            "heavy_hitter_ratio": args.heavy_hitter_ratio,
        },
        "results": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote baseline results to {output_path}")
    for method, method_data in all_results.items():
        print(f"[{method}] {json.dumps(method_data['summary'], indent=2)}")


if __name__ == "__main__":
    main()
