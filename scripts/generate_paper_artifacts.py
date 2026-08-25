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
from benchmarks.io_utils import write_json_atomic


def resolve_input_paths(
    *,
    inputs: list[str],
    suite_manifest: str | None,
    job_prefix: str | None,
    allow_glob: bool,
) -> list[str]:
    paths: list[str] = []
    for pattern in inputs:
        if glob.has_magic(pattern) and not allow_glob:
            raise ValueError(
                "Wildcard discovery is disabled for paper artifacts; pass the "
                "suite manifest's explicit output paths."
            )
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    if suite_manifest:
        manifest_path = Path(suite_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for job in manifest.get("jobs", []):
            if job.get("status") not in {"complete", "skipped_complete"}:
                continue
            if job_prefix and not str(job.get("name", "")).startswith(job_prefix):
                continue
            output = Path(str(job.get("output", "")))
            if not output.is_absolute():
                output = manifest_path.parent / output
            paths.append(str(output.resolve()))
    paths = [
        path
        for path in dict.fromkeys(paths)
        if not path.endswith((".checkpoint.json", ".checkpoint.sqlite"))
        and not path.endswith("suite_manifest.json")
    ]
    if not paths:
        raise ValueError("No result JSON files were selected.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help="Result JSON paths or glob patterns",
    )
    parser.add_argument(
        "--suite-manifest",
        help="Select completed outputs from an explicit suite manifest",
    )
    parser.add_argument(
        "--job-prefix",
        help="With --suite-manifest, include only job names with this prefix",
    )
    parser.add_argument("--output-dir", default="outputs/paper/artifacts")
    parser.add_argument(
        "--allow-glob",
        action="store_true",
        help="Diagnostic-only: permit wildcard input discovery",
    )
    args = parser.parse_args()

    if not args.inputs and not args.suite_manifest:
        parser.error("provide --inputs and/or --suite-manifest")
    paths = resolve_input_paths(
        inputs=args.inputs,
        suite_manifest=args.suite_manifest,
        job_prefix=args.job_prefix,
        allow_glob=args.allow_glob,
    )
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
    approximate_methods = sorted(
        {
            str(row["method"])
            for row in rows
            if row.get("reference_equivalence") == "approximation"
        }
    )
    write_csv(rows, output / "all_results.csv")
    write_csv(paired_rows, output / "paired_significance.csv")
    write_json_atomic(output / "all_results.json", rows)
    write_json_atomic(
        output / "artifact_manifest.json",
        {
            "inputs": input_manifest,
            "aggregate_rows": len(rows),
            "paired_comparisons": len(paired_rows),
            "approximate_baseline_methods": approximate_methods,
            "claim_policy": (
                "Methods listed in approximate_baseline_methods are controlled "
                "local ports, not official-paper reproductions."
            ),
        },
    )
    write_markdown_tables(rows, output)
    generate_figures(rows, output / "figures")
    print(f"Loaded {len(runs)} successful runs from {len(paths)} files.")
    print(f"Generated {len(rows)} aggregate rows in {output}.")
    if approximate_methods:
        print(
            "Approximate baseline implementations (label accordingly): "
            + ", ".join(approximate_methods)
        )


if __name__ == "__main__":
    main()
