"""Benchmark runner for NIAH cache traces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.dataset_runner import run_dataset_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=str, default="data/niah_trace.jsonl")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--recent-window", type=int, default=16)
    parser.add_argument("--output", type=str, default="outputs/bench_niah.json")
    args = parser.parse_args()

    payload = run_dataset_benchmark(
        dataset_name="niah",
        trace_path=args.trace_path,
        output_path=args.output,
        budget=args.budget,
        theta=args.theta,
        recent_window=args.recent_window,
    )
    print(f"Wrote NIAH benchmark to {args.output}")
    print(payload["cache_summary"])


if __name__ == "__main__":
    main()
