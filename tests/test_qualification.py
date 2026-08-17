from benchmarks.qualification import qualification_report


def _payload():
    return {
        "experiment_fingerprint": "fingerprint",
        "environment": {
            "cuda": {"available": True, "driver_version": "999.1"},
            "git": {"commit": "abc123", "dirty": False},
        },
        "models": ["model"],
        "model_revisions": {"model": "abc123"},
        "requested_model_revisions": {"model": "abc123"},
        "grid": {
            "max_new_tokens": 2,
            "min_budget_utilization": 0.99,
            "max_budget_shortfall_tokens": 1,
        },
        "summary": {"fullkv_parity": {"all_passed": True}},
        "runs": [
            {
                "status": "ok",
                "method": "tdc_kv",
                "model": "model",
                "dataset": "data",
                "sample_id": "1",
                "run_key": "key",
                "generated_token_ids": [1, 2],
                "prompt_truncated": False,
                "runtime": {"stages": {"policy": {"elapsed_ms": 1.0}}},
                "cache_memory": {
                    "kv_bytes_before": 100,
                    "kv_bytes_after": 50,
                    "kv_bytes_saved": 50,
                },
                "metrics": {
                    "budget_overflow": 0,
                    "budget_shortfall": 0,
                    "budget_utilization": 1.0,
                },
            }
        ],
    }


def test_qualification_report_accepts_complete_result():
    report = qualification_report(_payload(), require_parity=True, require_cuda=True)
    assert report["passed"] is True
    assert report["require_parity"] is True
    assert report["require_cuda"] is True
    assert report["failures"] == []


def test_qualification_report_lists_paper_blockers():
    payload = _payload()
    payload["environment"]["cuda"]["available"] = False
    payload["summary"]["fullkv_parity"]["all_passed"] = False
    payload["runs"][0]["generated_token_ids"] = []
    payload["runs"][0]["metrics"].update(
        budget_overflow=1,
        budget_shortfall=2,
        budget_utilization=0.5,
    )
    report = qualification_report(payload, require_parity=True, require_cuda=True)
    assert report["passed"] is False
    assert len(report["failures"]) == 6


def test_strict_qualification_rejects_truncation_and_unpinned_model():
    payload = _payload()
    payload["runs"][0]["prompt_truncated"] = True
    payload["requested_model_revisions"] = {}

    report = qualification_report(
        payload,
        require_no_truncation=True,
        require_model_revision=True,
    )

    assert report["passed"] is False
    assert any("truncated prompt" in failure for failure in report["failures"])
    assert any("revision pins" in failure for failure in report["failures"])
