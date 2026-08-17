"""Frozen dataset selection manifests for leakage-free paper evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SELECTION_OPTION_KEYS = {"manifest", "sample_manifest", "partition"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def record_sha256(record: dict[str, Any]) -> str:
    source_record = {
        key: value
        for key, value in record.items()
        if not str(key).startswith("__tdc_")
    }
    return hashlib.sha256(_canonical_json(source_record).encode("utf-8")).hexdigest()


def record_identity(record: dict[str, Any], *, id_field: str | None) -> str:
    if id_field and record.get(id_field) is not None:
        return str(record[id_field])
    return f"sha256:{record_sha256(record)}"


def dataset_signature(spec: Any) -> dict[str, Any]:
    payload = spec.to_dict()
    options = dict(payload.get("options") or {})
    payload["options"] = {
        key: value for key, value in options.items() if key not in SELECTION_OPTION_KEYS
    }
    return payload


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_frozen_manifest(
    *,
    specs: Iterable[Any],
    records_by_dataset: dict[str, list[dict[str, Any]]],
    partition_sizes: dict[str, int],
) -> dict[str, Any]:
    """Build disjoint, sequential partitions with record-level hashes."""
    if not partition_sizes:
        raise ValueError("At least one dataset partition is required.")
    normalized_sizes = {
        str(name): max(0, int(size)) for name, size in partition_sizes.items()
    }
    total_required = sum(normalized_sizes.values())
    datasets: dict[str, Any] = {}
    for spec in specs:
        records = records_by_dataset.get(spec.name, [])
        if len(records) < total_required:
            raise ValueError(
                f"Dataset `{spec.name}` has {len(records)} records but "
                f"{total_required} are required by the frozen partitions."
            )
        cursor = 0
        partitions: dict[str, list[dict[str, Any]]] = {}
        for partition, size in normalized_sizes.items():
            selectors = []
            for index in range(cursor, cursor + size):
                record = records[index]
                selectors.append(
                    {
                        "index": index,
                        "identity": record_identity(record, id_field=spec.id_field),
                        "sha256": record_sha256(record),
                    }
                )
            partitions[partition] = selectors
            cursor += size
        datasets[spec.name] = {
            "signature": dataset_signature(spec),
            "partitions": partitions,
        }
    return {
        "schema_version": 1,
        "selection_policy": "disjoint_sequential_record_hashes",
        "partition_sizes": normalized_sizes,
        "datasets": datasets,
    }


def write_frozen_manifest(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _manifest_entry(
    *,
    path: str | Path,
    spec: Any,
    partition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = (payload.get("datasets") or {}).get(spec.name)
    if not isinstance(entry, dict):
        raise KeyError(f"Frozen manifest has no dataset named `{spec.name}`.")
    if entry.get("signature") != dataset_signature(spec):
        raise ValueError(
            f"Frozen manifest signature does not match dataset `{spec.name}`."
        )
    selectors = (entry.get("partitions") or {}).get(str(partition))
    if not isinstance(selectors, list):
        raise KeyError(
            f"Frozen manifest dataset `{spec.name}` has no `{partition}` partition."
        )
    return payload, selectors


def required_record_count(
    *,
    path: str | Path,
    spec: Any,
    partition: str,
) -> int:
    _, selectors = _manifest_entry(path=path, spec=spec, partition=partition)
    return max((int(selector["index"]) for selector in selectors), default=-1) + 1


def select_frozen_records(
    records: list[dict[str, Any]],
    *,
    path: str | Path,
    spec: Any,
    partition: str,
) -> list[dict[str, Any]]:
    """Select and verify an immutable record partition."""
    _, selectors = _manifest_entry(path=path, spec=spec, partition=partition)
    selected: list[dict[str, Any]] = []
    for selector in selectors:
        index = int(selector["index"])
        if index < 0 or index >= len(records):
            raise IndexError(
                f"Frozen selector index {index} is unavailable for `{spec.name}`."
            )
        record = records[index]
        identity = record_identity(record, id_field=spec.id_field)
        digest = record_sha256(record)
        if identity != str(selector.get("identity")) or digest != selector.get("sha256"):
            raise ValueError(
                f"Dataset `{spec.name}` record {index} changed after the manifest "
                "was frozen. Pin the dataset revision or rebuild the protocol."
            )
        copied = dict(record)
        copied["__tdc_source_index"] = index
        copied["__tdc_frozen_identity"] = identity
        selected.append(copied)
    return selected


__all__ = [
    "build_frozen_manifest",
    "dataset_signature",
    "manifest_sha256",
    "record_identity",
    "record_sha256",
    "required_record_count",
    "select_frozen_records",
    "write_frozen_manifest",
]
