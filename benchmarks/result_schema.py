"""Schema-v2 validation for experiment rows and result artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from benchmarks.numerical_validation import find_nonfinite_paths


SCHEMA_VERSION = 2
RESULT_STATES = {"in_progress", "complete", "failed"}
RUN_STATES = {"ok", "error"}


class ResultValidationError(ValueError):
    def __init__(self, failures: list[str]) -> None:
        self.failures = list(failures)
        super().__init__("; ".join(self.failures))


def validate_run_row(row: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    status = row.get("status")
    if status not in RUN_STATES:
        failures.append(f"invalid run status {status!r}")
        return failures
    nonfinite = find_nonfinite_paths(dict(row))
    if nonfinite:
        failures.append(f"run contains non-finite values at {nonfinite[:5]}")
    if status == "ok":
        for field in (
            "run_key",
            "execution_contract_id",
            "sample_input_id",
            "method_config_id",
        ):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64:
                failures.append(f"successful run has invalid `{field}`")
        numerical = row.get("numerical_health")
        if not isinstance(numerical, dict) or numerical.get("passed") is not True:
            failures.append("successful run lacks passing numerical health")
        if not isinstance(row.get("generated_token_ids"), list):
            failures.append("successful run lacks generated_token_ids list")
        if not isinstance(row.get("generation_health"), dict):
            failures.append("successful run lacks generation_health")
    else:
        if not row.get("error_type") or not row.get("error"):
            failures.append("error run lacks error_type/error detail")
    return failures


def validate_result_payload(
    payload: Mapping[str, Any],
    *,
    require_complete: bool = False,
) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append(
            f"schema_version must be {SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    state = payload.get("state")
    if state not in RESULT_STATES:
        failures.append(f"invalid result state {state!r}")
    if require_complete and state != "complete":
        failures.append("result state is not complete")
    for field in ("experiment_fingerprint", "job_fingerprint", "protocol_fingerprint"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            failures.append(f"invalid `{field}`")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        failures.append("runs must be a list")
        return failures
    keys: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(runs):
        if not isinstance(row, Mapping):
            failures.append(f"run[{index}] is not an object")
            continue
        failures.extend(f"run[{index}]: {failure}" for failure in validate_run_row(row))
        key = row.get("run_key")
        if isinstance(key, str):
            if key in keys and dict(keys[key]) != dict(row):
                failures.append(f"conflicting duplicate run_key {key}")
            keys[key] = row

    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        failures.append("coverage is missing")
    else:
        successful = sum(1 for row in runs if row.get("status") == "ok")
        failed = sum(1 for row in runs if row.get("status") == "error")
        observed = len(runs)
        expected_counts = {
            "observed_runs": observed,
            "successful_runs": successful,
            "failed_runs": failed,
            "validated_runs": successful,
        }
        for field, expected in expected_counts.items():
            if coverage.get(field) != expected:
                failures.append(
                    f"coverage `{field}` is {coverage.get(field)!r}, expected {expected}"
                )
    return failures


def assert_result_payload(
    payload: Mapping[str, Any],
    *,
    require_complete: bool = False,
) -> None:
    failures = validate_result_payload(payload, require_complete=require_complete)
    if failures:
        raise ResultValidationError(failures)


__all__ = [
    "RESULT_STATES",
    "RUN_STATES",
    "SCHEMA_VERSION",
    "ResultValidationError",
    "assert_result_payload",
    "validate_result_payload",
    "validate_run_row",
]
