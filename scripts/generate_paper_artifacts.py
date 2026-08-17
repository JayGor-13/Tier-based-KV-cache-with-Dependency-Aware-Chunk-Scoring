"""Generate final CSV tables, Markdown tables, and figures from run JSON files."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.paper_reporting import (
    aggregate_paper_rows,
    generate_figures,
    load_paper_runs,
    paired_significance_rows,
    write_csv,
    write_markdown_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Result JSON paths or glob patterns",
    )
    parser.add_argument("--output-dir", default="outputs/paper/artifacts")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    paths = [
        path
        for path in dict.fromkeys(paths)
        if not path.endswith(".checkpoint.json")
        and not path.endswith("suite_manifest.json")
    ]
    if not paths:
        raise ValueError("No result JSON files matched --inputs.")
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing result files: {missing}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_manifest = []
    for path_value in paths:
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        qualification = payload.get("summary", {}).get("qualification", {})
        if qualification.get("passed") is not True:
            raise ValueError(
                f"{path} is not paper-qualified: {qualification.get('failures')}"
            )
        if int(payload.get("summary", {}).get("failed_runs", 0)) != 0:
            raise ValueError(f"{path} contains failed runs.")
        input_manifest.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "experiment_fingerprint": payload.get("experiment_fingerprint"),
                "seed": payload.get("environment", {}).get("seed"),
                "successful_runs": payload.get("summary", {}).get(
                    "successful_runs"
                ),
            }
        )
    runs = load_paper_runs(paths)
    rows = aggregate_paper_rows(runs)
    paired_rows = paired_significance_rows(runs)
    write_csv(rows, output / "all_results.csv")
    write_csv(paired_rows, output / "paired_significance.csv")
    (output / "all_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (output / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "inputs": input_manifest,
                "aggregate_rows": len(rows),
                "paired_comparisons": len(paired_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_tables(rows, output)
    generate_figures(rows, output / "figures")
    print(f"Loaded {len(runs)} successful runs from {len(paths)} files.")
    print(f"Generated {len(rows)} aggregate rows in {output}.")


if __name__ == "__main__":
    main()
