import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _selectors(path, dataset, partition):
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return payload, payload["datasets"][dataset]["partitions"][partition]


def test_diagnostic_manifest_has_three_unique_training_records():
    payload, selectors = _selectors(
        "protocol/gsm8k_diagnostic_manifest.json",
        "gsm8k_diagnostic",
        "diagnostic",
    )

    assert payload["datasets"]["gsm8k_diagnostic"]["signature"]["split"] == "train"
    assert [row["index"] for row in selectors] == [0, 1, 2]
    assert len({row["sha256"] for row in selectors}) == 3


def test_full_manifest_covers_every_official_gsm8k_test_index_once():
    payload, selectors = _selectors(
        "protocol/gsm8k_full_manifest.json",
        "gsm8k_full",
        "final",
    )

    assert payload["partition_sizes"] == {"final": 1319}
    assert payload["datasets"]["gsm8k_full"]["signature"]["split"] == "test"
    assert [row["index"] for row in selectors] == list(range(1319))
    assert len({row["sha256"] for row in selectors}) == 1319
