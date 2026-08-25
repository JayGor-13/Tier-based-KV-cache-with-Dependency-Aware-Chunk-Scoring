import csv
import json

from scripts.validate_results import audit_path, audit_results


def test_legacy_summary_is_numerically_audited_but_not_paper_eligible(tmp_path):
    path = tmp_path / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "dataset",
                "method",
                "samples",
                "runs",
                "quality_mean",
                "requested_retention_ratio",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "tiny",
                "dataset": "gsm8k",
                "method": "fullkv",
                "samples": 2,
                "runs": 2,
                "quality_mean": 0,
                "requested_retention_ratio": 1,
            }
        )

    report = audit_path(path)

    assert report["artifact_class"] == "legacy_summary"
    assert report["corrupt_records"] == 0
    assert report["paper_eligible"] is False
    assert any("every reported quality_mean is zero" in item for item in report["warnings"])


def test_json_audit_catches_nonfinite_and_degenerate_success_rows(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "status": "ok",
                        "run_key": "r",
                        "metrics": {"score": float("nan")},
                        "generated_token_ids": [0] * 8,
                        "config": {"max_new_tokens": 8},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_path(path)
    codes = {issue["code"] for issue in report["blockers"]}

    assert report["corrupt_records"] == 1
    assert {"nonfinite", "degenerate_generation"}.issubset(codes)


def test_missing_path_is_reported_without_mutation(tmp_path):
    path = tmp_path / "missing.json"
    report = audit_results([path])

    assert report["source_count"] == 1
    assert report["sources"][0]["exists"] is False
    assert not path.exists()
