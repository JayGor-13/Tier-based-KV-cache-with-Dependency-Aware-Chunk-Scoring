import json

import pytest

from benchmarks.dataset_manifests import (
    build_frozen_manifest,
    record_sha256,
    select_frozen_records,
    write_frozen_manifest,
)
from benchmarks.hf_runner import parse_dataset_spec


def test_frozen_partitions_are_disjoint_and_hash_verified(tmp_path):
    spec = parse_dataset_spec(
        "name=fixture,source=fixture.jsonl,prompt_field=question,id_field=id"
    )
    records = [
        {"id": f"sample-{index}", "question": f"Question {index}"}
        for index in range(8)
    ]
    payload = build_frozen_manifest(
        specs=[spec],
        records_by_dataset={"fixture": records},
        partition_sizes={"tuning": 3, "final": 5},
    )
    path = tmp_path / "manifest.json"
    write_frozen_manifest(payload, path)

    tuning = select_frozen_records(
        records, path=path, spec=spec, partition="tuning"
    )
    final = select_frozen_records(records, path=path, spec=spec, partition="final")

    assert {row["id"] for row in tuning}.isdisjoint(
        {row["id"] for row in final}
    )
    assert [row["__tdc_source_index"] for row in final] == [3, 4, 5, 6, 7]

    changed = list(records)
    changed[3] = {"id": "sample-3", "question": "Changed"}
    with pytest.raises(ValueError, match="changed after the manifest was frozen"):
        select_frozen_records(
            changed, path=path, spec=spec, partition="final"
        )


def test_manifest_file_is_valid_json(tmp_path):
    destination = tmp_path / "manifest.json"
    write_frozen_manifest({"schema_version": 1, "datasets": {}}, destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == 1


def test_internal_selection_metadata_does_not_change_source_record_hash():
    assert record_sha256({"id": "a", "value": 1}) == record_sha256(
        {"id": "a", "value": 1, "__tdc_source_index": 7}
    )
