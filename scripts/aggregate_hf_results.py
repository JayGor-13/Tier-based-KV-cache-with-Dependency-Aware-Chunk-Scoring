"""Create grouped model/dataset/method/configuration summaries from HF runs."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import aggregate_grouped_runs
from benchmarks.io_utils import write_json_atomic
from benchmarks.result_schema import assert_result_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate raw HuggingFace experiment runs"
    )
    parser.add_argument("--input", required=True, help="HF result JSON containing runs")
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path; defaults to <input_stem>_grouped.json",
    )
    parser.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="Diagnostic-only: aggregate legacy/unqualified inputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not args.allow_unqualified:
        assert_result_payload(payload, require_complete=True)
        if (payload.get("summary") or {}).get("qualification", {}).get(
            "passed"
        ) is not True:
            raise ValueError(f"{input_path} is not paper-qualified")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"{input_path} does not contain a `runs` list.")

    grouped_results = aggregate_grouped_runs(runs)
    output_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path),
        "group_count": len(grouped_results),
        "grouped_results": grouped_results,
    }
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_grouped.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, output_payload)
    print(f"Grouped {len(runs)} runs into {len(grouped_results)} result groups.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
