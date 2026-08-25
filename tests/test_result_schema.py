from copy import deepcopy

from benchmarks.result_schema import assert_result_payload, validate_result_payload


def _row():
    return {
        "status": "ok",
        "run_key": "a" * 64,
        "execution_contract_id": "b" * 64,
        "sample_input_id": "c" * 64,
        "method_config_id": "d" * 64,
        "generated_token_ids": [1, 2],
        "generation_health": {"degenerate_repetition": False},
        "numerical_health": {"passed": True},
    }


def _payload():
    return {
        "schema_version": 2,
        "state": "complete",
        "experiment_fingerprint": "e" * 64,
        "job_fingerprint": "e" * 64,
        "protocol_fingerprint": "f" * 64,
        "coverage": {
            "observed_runs": 1,
            "successful_runs": 1,
            "failed_runs": 0,
            "validated_runs": 1,
        },
        "runs": [_row()],
    }


def test_complete_schema_v2_payload_passes():
    assert_result_payload(_payload(), require_complete=True)


def test_schema_rejects_nonfinite_success_and_bad_coverage():
    payload = _payload()
    payload["runs"][0]["score"] = float("nan")
    payload["coverage"]["successful_runs"] = 2

    failures = validate_result_payload(payload, require_complete=True)

    assert any("non-finite" in failure for failure in failures)
    assert any("successful_runs" in failure for failure in failures)


def test_schema_rejects_conflicting_duplicate_run_key():
    payload = _payload()
    duplicate = deepcopy(payload["runs"][0])
    duplicate["generated_token_ids"] = [9]
    payload["runs"].append(duplicate)
    payload["coverage"].update(
        observed_runs=2,
        successful_runs=2,
        validated_runs=2,
    )

    failures = validate_result_payload(payload)

    assert any("conflicting duplicate" in failure for failure in failures)
