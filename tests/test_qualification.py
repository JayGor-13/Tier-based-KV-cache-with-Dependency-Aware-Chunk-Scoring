from benchmarks.qualification import qualification_report, recompute_declared_qualification


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
                "generation_health": {
                    "decoded_nonempty": True,
                    "token_count": 2,
                    "unique_token_count": 2,
                    "unique_non_special_token_count": 2,
                    "maximum_non_special_token_fraction": 0.5,
                    "reached_generation_limit": True,
                    "degenerate_repetition": False,
                },
                "numerical_health": {
                    "passed": True,
                    "nonfinite_tensor_count": 0,
                },
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


def test_qualification_rejects_corrupted_nan_repeated_token_pattern():
    payload = _payload()
    run = payload["runs"][0]
    run["generated_token_ids"] = [0] * 96
    run["generation_health"] = {
        "decoded_nonempty": True,
        "token_count": 96,
        "unique_token_count": 1,
        "unique_non_special_token_count": 1,
        "maximum_non_special_token_fraction": 1.0,
        "reached_generation_limit": True,
        "degenerate_repetition": True,
    }
    run["score_min"] = float("nan")
    run["numerical_health"] = {
        "passed": False,
        "first_failure_stage": "prefill_argmax",
        "nonfinite_tensor_count": 1,
    }
    payload["grid"]["max_new_tokens"] = 96

    report = qualification_report(payload)

    assert report["passed"] is False
    assert any("non-finite" in failure for failure in report["failures"])
    assert any("numerical-health" in failure for failure in report["failures"])
    assert any("degenerate" in failure for failure in report["failures"])


def test_qualification_requires_declared_parity_coverage():
    payload = _payload()
    payload["parity_records"] = [
        {"token_match": True, "text_match": True},
    ]

    report = qualification_report(
        payload,
        require_parity=True,
        expected_parity_records=10,
    )

    assert report["passed"] is False
    assert any("coverage mismatch" in failure for failure in report["failures"])


def test_qualification_reports_malformed_runtime_and_budget_without_crashing():
    payload = _payload()
    payload["runs"][0]["runtime"]["stages"]["policy"]["elapsed_ms"] = "broken"
    payload["runs"][0]["metrics"]["budget_overflow"] = "broken"

    report = qualification_report(payload)

    assert report["passed"] is False
    assert any("invalid runtime" in failure for failure in report["failures"])
    assert any("malformed budget" in failure for failure in report["failures"])


def test_declared_qualification_is_recomputed_instead_of_trusting_passed_flag():
    payload = _payload()
    payload["summary"]["qualification"] = qualification_report(payload)
    assert payload["summary"]["qualification"]["passed"] is True
    payload["runs"][0]["metrics"]["budget_overflow"] = 3

    recomputed = recompute_declared_qualification(payload)

    assert recomputed["stored_passed"] is True
    assert recomputed["passed"] is False
    assert any("exceeds the KV budget" in failure for failure in recomputed["failures"])
