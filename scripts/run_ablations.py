"""Run TDC-KV ablations for theta/recent-window settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import summarize_cache_metrics
from benchmarks.pipeline import load_trace_samples, run_tdc_policy


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=str, required=True)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument(
        "--theta-grid",
        type=str,
        default="0.2,0.3,0.4",
        help="Comma-separated theta values.",
    )
    parser.add_argument(
        "--recent-window-grid",
        type=str,
        default="8,16,32",
        help="Comma-separated recent window values.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ablations_results.json",
        help="Output JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    theta_grid = _parse_csv_floats(args.theta_grid)
    recent_window_grid = _parse_csv_ints(args.recent_window_grid)
    samples = load_trace_samples(args.trace_path)

    ablation_rows = []
    for theta in theta_grid:
        for recent_window in recent_window_grid:
            metrics = []
            for sample in samples:
                budget = int(args.budget if args.budget is not None else (sample.budget or 0))
                if budget <= 0:
                    raise ValueError(
                        f"No valid budget provided for sample `{sample.sample_id}`. "
                        "Use --budget or include `budget` in trace records."
                    )
                _, _, metric = run_tdc_policy(
                    sample,
                    budget=budget,
                    theta=theta,
                    recent_window=recent_window,
                )
                metrics.append(metric)

            summary = summarize_cache_metrics(metrics)
            ablation_rows.append(
                {
                    "theta": theta,
                    "recent_window": recent_window,
                    "summary": summary,
                }
            )

    ablation_rows.sort(key=lambda x: x["summary"]["avg_compression_ratio"], reverse=True)
    payload = {
        "config": {
            "trace_path": args.trace_path,
            "budget": args.budget,
            "theta_grid": theta_grid,
            "recent_window_grid": recent_window_grid,
        },
        "ablations": ablation_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ablation results to {output_path}")
    if ablation_rows:
        best = ablation_rows[0]
        print(
            "Best config by avg_compression_ratio: "
            f"theta={best['theta']}, recent_window={best['recent_window']}"
        )


if __name__ == "__main__":
    main()
