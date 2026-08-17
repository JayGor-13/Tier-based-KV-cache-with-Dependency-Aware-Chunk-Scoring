"""Freeze disjoint qualification, tuning, and final dataset partitions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.dataset_manifests import build_frozen_manifest, write_frozen_manifest
from benchmarks.hf_runner import load_dataset_records, parse_dataset_spec


def _parse_partitions(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in text.split(","):
        name, separator, value = item.strip().partition("=")
        if not separator or not name:
            raise ValueError("Partitions must use name=count entries.")
        result[name] = int(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        required=True,
        help="Semicolon-separated dataset specifications",
    )
    parser.add_argument(
        "--partitions",
        default="qualification=5,tuning=50,final=200",
    )
    parser.add_argument(
        "--output",
        default="protocol/paper_dataset_manifest.json",
    )
    args = parser.parse_args()

    specs = [
        parse_dataset_spec(value.strip())
        for value in args.datasets.split(";")
        if value.strip()
    ]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Every frozen dataset specification must have a unique name.")
    partitions = _parse_partitions(args.partitions)
    required = sum(partitions.values())
    records = {
        spec.name: load_dataset_records(spec, max_samples=required) for spec in specs
    }
    manifest = build_frozen_manifest(
        specs=specs,
        records_by_dataset=records,
        partition_sizes=partitions,
    )
    write_frozen_manifest(manifest, args.output)
    print(f"Frozen {len(specs)} datasets to {Path(args.output).resolve()}.")


if __name__ == "__main__":
    main()
