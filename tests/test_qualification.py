from benchmarks.qualification import qualification_report


def _payload():
    return {
        "experiment_fingerprint": "fingerprint",
        "environment": {
            "cuda": {"available": True},
            "git": {"commit": "abc123"},
        },
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
                "runtime": {"stages": {"policy": {"elapsed_ms": 1.0}}},
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
    assert report == {
        "passed": True,
        "require_parity": True,
        "require_cuda": True,
        "failures": [],
    }


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
