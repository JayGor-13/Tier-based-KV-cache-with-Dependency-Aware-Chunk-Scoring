import json

import pytest

from scripts.generate_paper_artifacts import resolve_input_paths


def test_suite_manifest_selects_only_completed_matching_jobs(tmp_path):
    main = tmp_path / "main_one.json"
    timing = tmp_path / "timing_one.json"
    main.write_text("{}", encoding="utf-8")
    timing.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "suite_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": "main_one", "output": str(main), "status": "complete"},
                    {"name": "main_failed", "output": "missing.json", "status": "failed"},
                    {"name": "timing_one", "output": str(timing), "status": "complete"},
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = resolve_input_paths(
        inputs=[],
        suite_manifest=str(manifest),
        job_prefix="main_",
        allow_glob=False,
    )

    assert selected == [str(main.resolve())]


def test_wildcards_remain_disabled_by_default():
    with pytest.raises(ValueError, match="Wildcard discovery is disabled"):
        resolve_input_paths(
            inputs=["outputs/*.json"],
            suite_manifest=None,
            job_prefix=None,
            allow_glob=False,
        )
