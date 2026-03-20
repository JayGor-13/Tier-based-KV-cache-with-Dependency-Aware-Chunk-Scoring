"""Run main TDC-KV results from precomputed trace inputs."""

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=str, required=True, help="JSON/JSONL/PT trace input.")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Global cache budget B. If omitted, uses per-sample budget in trace.",
    )
    parser.add_argument("--theta", type=float, default=0.3, help="Top-ratio for Tier-1 masking.")
    parser.add_argument(
        "--recent-window",
        type=int,
        default=16,
        help="Recent token count for Tier-2 masking.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/main_results.json",
        help="Output JSON file.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    samples = load_trace_samples(args.trace_path)

    run_rows = []
    metric_rows = []
    for sample in samples:
        budget = int(args.budget if args.budget is not None else (sample.budget or 0))
        if budget <= 0:
            raise ValueError(
                f"No valid budget provided for sample `{sample.sample_id}`. "
                "Use --budget or include `budget` in trace records."
            )

        result, tiers, metrics = run_tdc_policy(
            sample,
            budget=budget,
            theta=args.theta,
            recent_window=args.recent_window,
        )
        metric_rows.append(metrics)
        run_rows.append(
            {
                "sample_id": sample.sample_id,
                "budget": budget,
                "kept_tokens": int(result.kept_indices.numel()),
                "removed_tokens": int(result.removed_indices.numel()),
                "tier0_chunks": int((tiers == 0).sum().item()),
                "tier1_chunks": int((tiers == 1).sum().item()),
                "tier2_chunks": int((tiers == 2).sum().item()),
                "metrics": metrics.to_dict(),
            }
        )

    summary = summarize_cache_metrics(metric_rows)
    payload = {
        "config": {
            "trace_path": args.trace_path,
            "budget": args.budget,
            "theta": args.theta,
            "recent_window": args.recent_window,
            "method": "tdc_kv",
        },
        "summary": summary,
        "runs": run_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote main results to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
