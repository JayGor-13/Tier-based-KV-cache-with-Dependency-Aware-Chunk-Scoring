"""Hard qualification checks for paper-grade experiment outputs."""

from __future__ import annotations

import math
from typing import Any

from benchmarks.numerical_validation import find_nonfinite_paths, generation_health


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _integer(value: Any) -> int | None:
    converted = _finite_float(value)
    if converted is None or not converted.is_integer():
        return None
    return int(converted)


def _descriptor_matches(
    descriptor: Any,
    *,
    budget_type: str,
    budget_value: float | int,
) -> bool:
    if not isinstance(descriptor, dict) or descriptor.get("type") != budget_type:
        return False
    observed = _finite_float(descriptor.get("value"))
    return observed is not None and abs(observed - float(budget_value)) < 1e-12


def qualification_failures(
    result: dict[str, Any],
    *,
    require_parity: bool = True,
    require_cuda: bool = False,
    require_frozen_manifest: bool = False,
    require_model_revision: bool = False,
    require_driver: bool = False,
    require_clean_git: bool = False,
    require_preflight: bool = False,
    require_exact_niah: bool = False,
    require_no_truncation: bool = False,
    require_numerical_health: bool = True,
    require_generation_health: bool = True,
    require_fullkv_pairing: bool = False,
    expected_parity_records: int | None = None,
    min_gsm8k_parse_rate: float = 0.0,
) -> list[str]:
    """Return actionable reasons an experiment output is not paper-qualified."""
    failures: list[str] = []
    runs = result.get("runs")
    if not isinstance(runs, list) or not runs:
        return ["no experiment runs were recorded"]

    error_rows = [row for row in runs if row.get("status") != "ok"]
    if error_rows:
        failures.append(f"{len(error_rows)} run(s) failed")

    successful = [row for row in runs if row.get("status") == "ok"]
    if not successful:
        failures.append("no successful runs were recorded")
    parsed_min_parse_rate = _finite_float(min_gsm8k_parse_rate)
    if parsed_min_parse_rate is None or not 0.0 <= parsed_min_parse_rate <= 1.0:
        raise ValueError("min_gsm8k_parse_rate must be between 0 and 1.")

    grid = result.get("grid", {})
    if not isinstance(grid, dict):
        failures.append("experiment grid metadata is not an object")
        grid = {}
    max_new_tokens = _integer(grid.get("max_new_tokens", 0))
    min_utilization = _finite_float(grid.get("min_budget_utilization", 0.99))
    max_shortfall = _integer(grid.get("max_budget_shortfall_tokens", 1))
    if max_new_tokens is None or max_new_tokens < 0:
        failures.append("grid max_new_tokens is invalid")
        max_new_tokens = 0
    if min_utilization is None or not 0.0 <= min_utilization <= 1.0:
        failures.append("grid min_budget_utilization is invalid")
        min_utilization = 0.99
    if max_shortfall is None or max_shortfall < 0:
        failures.append("grid max_budget_shortfall_tokens is invalid")
        max_shortfall = 1

    def parsed_grid_values(field: str, converter: Any) -> list[Any]:
        raw_values = grid.get(field, [])
        if not isinstance(raw_values, list):
            failures.append(f"grid {field} is not a list")
            return []
        parsed: list[Any] = []
        for value in raw_values:
            converted = converter(value)
            if converted is None:
                failures.append(f"grid {field} contains invalid value {value!r}")
                continue
            parsed.append(converted)
        return parsed

    grid_ratios = parsed_grid_values("budget_ratios", _finite_float)
    grid_budgets = parsed_grid_values("budgets", _integer)
    grid_thetas = parsed_grid_values("thetas", _finite_float)
    grid_windows = parsed_grid_values("recent_windows", _integer)
    grid_alphas = parsed_grid_values("alphas", _finite_float)
    run_keys = [str(row.get("run_key")) for row in successful if row.get("run_key")]
    if len(run_keys) != len(set(run_keys)):
        failures.append("successful runs contain duplicate deterministic run keys")

    for index, row in enumerate(successful):
        identity = (
            f"run[{index}] {row.get('model')}/{row.get('dataset')}/"
            f"{row.get('sample_id')}/{row.get('method')}"
        )
        if not row.get("run_key"):
            failures.append(f"{identity} is missing its deterministic run key")
        if max_new_tokens > 0 and not row.get("generated_token_ids"):
            failures.append(f"{identity} generated no tokens")
        nonfinite_paths = find_nonfinite_paths(row)
        if nonfinite_paths:
            preview = nonfinite_paths[:5]
            failures.append(
                f"{identity} contains non-finite numeric values at {preview}"
            )
        numerical = row.get("numerical_health")
        if require_numerical_health and (
            not isinstance(numerical, dict) or numerical.get("passed") is not True
        ):
            failures.append(f"{identity} has no passing numerical-health record")
        token_ids = row.get("generated_token_ids") or []
        observed_generation_health = row.get("generation_health")
        if not isinstance(observed_generation_health, dict):
            observed_generation_health = generation_health(
                token_ids,
                max_new_tokens=max_new_tokens,
            )
        if require_generation_health and observed_generation_health.get(
            "degenerate_repetition"
        ) is True:
            failures.append(f"{identity} has degenerate repeated-token generation")
        if require_no_truncation and row.get("prompt_truncated") is not False:
            failures.append(f"{identity} used a truncated prompt")

        runtime = row.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("stages"):
            failures.append(f"{identity} has no stage runtime measurements")
        else:
            stages = runtime.get("stages")
            invalid_runtime = not isinstance(stages, dict) or not stages
            if isinstance(stages, dict):
                for stage in stages.values():
                    elapsed = (
                        _finite_float(stage.get("elapsed_ms"))
                        if isinstance(stage, dict)
                        else None
                    )
                    if elapsed is None or elapsed < 0.0:
                        invalid_runtime = True
                        break
            if invalid_runtime:
                failures.append(f"{identity} has an invalid runtime measurement")

        cache_memory = row.get("cache_memory") or {}
        for field in ("kv_bytes_before", "kv_bytes_after", "kv_bytes_saved"):
            if field not in cache_memory:
                failures.append(f"{identity} is missing cache-memory field `{field}`")
        if all(
            isinstance(cache_memory.get(field), int)
            for field in ("kv_bytes_before", "kv_bytes_after", "kv_bytes_saved")
        ):
            before = int(cache_memory["kv_bytes_before"])
            after = int(cache_memory["kv_bytes_after"])
            saved = int(cache_memory["kv_bytes_saved"])
            if min(before, after, saved) < 0 or before - after != saved:
                failures.append(f"{identity} has inconsistent cache-memory arithmetic")

        if row.get("method") == "fullkv":
            metrics = row.get("metrics") or {}
        else:
            metrics = row.get("metrics") or {}
            overflow = _integer(metrics.get("budget_overflow", 0))
            shortfall = _integer(metrics.get("budget_shortfall", 0))
            utilization = _finite_float(metrics.get("budget_utilization", 0.0))
            if overflow is None or shortfall is None or utilization is None:
                failures.append(f"{identity} has malformed budget metrics")
                overflow = max(0, overflow or 0)
                shortfall = max(0, shortfall or 0)
                utilization = utilization if utilization is not None else 0.0
            if overflow > 0:
                failures.append(f"{identity} exceeds the KV budget by {overflow} token(s)")
            if shortfall > max_shortfall:
                failures.append(
                    f"{identity} underfills the KV budget by {shortfall} token(s)"
                )
            if utilization < min_utilization:
                failures.append(
                    f"{identity} budget utilization {utilization:.6f} is below "
                    f"{min_utilization:.6f}"
                )

        dataset_name = str(row.get("dataset", "")).lower()
        if "niah" in dataset_name or "needle" in dataset_name or "hotpot" in dataset_name:
            structural = row.get("structural_metrics") or {}
            critical_count = _integer(structural.get("critical_token_count", 0))
            if critical_count is None or critical_count <= 0:
                failures.append(
                    f"{identity} localized no answer-critical evidence tokens"
                )
            localization = structural.get("evidence_localization_rate")
            localization_value = _finite_float(localization)
            if localization_value is None or localization_value < 1.0:
                failures.append(
                    f"{identity} did not localize every declared evidence span"
                )
        if require_exact_niah and ("niah" in dataset_name or "needle" in dataset_name):
            metadata = row.get("dataset_runtime_metadata") or {}
            if metadata.get("tokenizer_exact") is not True:
                failures.append(f"{identity} did not use tokenizer-exact NIAH")
            if metadata.get("actual_context_tokens") != metadata.get(
                "target_context_tokens"
            ):
                failures.append(f"{identity} missed the requested NIAH token length")

    sample_manifest = result.get("sample_manifest") or {}
    if not isinstance(sample_manifest, dict):
        failures.append("sample_manifest is not an object")
        sample_manifest = {}
    raw_methods = grid.get("methods", [])
    if not isinstance(raw_methods, list):
        failures.append("grid methods is not a list")
        raw_methods = []
    requested_methods = [str(method) for method in raw_methods]
    raw_models = result.get("models", [])
    if not isinstance(raw_models, list):
        failures.append("models is not a list")
        raw_models = []
    for model in raw_models:
        for dataset, samples in sample_manifest.items():
            if not isinstance(samples, list):
                failures.append(f"sample_manifest[{dataset!r}] is not a list")
                continue
            for sample in samples:
                if not isinstance(sample, dict):
                    failures.append(f"sample_manifest[{dataset!r}] contains a non-object")
                    continue
                sample_id = str(sample.get("sample_id"))
                sample_rows = [
                    row
                    for row in successful
                    if str(row.get("model")) == str(model)
                    and str(row.get("dataset")) == str(dataset)
                    and str(row.get("sample_id")) == sample_id
                ]
                available_methods = {str(row.get("method")) for row in sample_rows}
                missing_methods = sorted(set(requested_methods) - available_methods)
                if missing_methods:
                    failures.append(
                        f"{model}/{dataset}/{sample_id} is missing methods "
                        f"{missing_methods}"
                    )
                for method in requested_methods:
                    if method == "fullkv":
                        continue
                    method_rows = [
                        row for row in sample_rows if str(row.get("method")) == method
                    ]
                    descriptors = [
                        specification
                        for row in method_rows
                        for specification in (
                            (row.get("config") or {}).get("budget_specifications") or []
                        )
                    ]
                    for ratio_value in grid_ratios:
                        if not any(
                            _descriptor_matches(
                                descriptor,
                                budget_type="ratio",
                                budget_value=ratio_value,
                            )
                            for descriptor in descriptors
                        ):
                            failures.append(
                                f"{model}/{dataset}/{sample_id}/{method} is missing "
                                f"budget ratio {ratio_value}"
                            )
                    for budget_value in grid_budgets:
                        if not any(
                            _descriptor_matches(
                                descriptor,
                                budget_type="absolute",
                                budget_value=budget_value,
                            )
                            for descriptor in descriptors
                        ):
                            failures.append(
                                f"{model}/{dataset}/{sample_id}/{method} is missing "
                                f"absolute budget {budget_value}"
                            )
                    if method == "tdc_kv":
                        parameter_points = [
                            (theta, window, alpha)
                            for theta in grid_thetas
                            for window in grid_windows
                            for alpha in grid_alphas
                        ]
                    elif method == "streamingllm":
                        parameter_points = [(None, None, None)]
                    else:
                        parameter_points = [
                            (None, window, None)
                            for window in grid_windows
                        ]
                    expected_budget_specs = [
                            ("ratio", value)
                            for value in grid_ratios
                        ] + [
                        ("absolute", value) for value in grid_budgets
                    ]
                    for theta, window, alpha in parameter_points:
                        matching_rows = []
                        for row in method_rows:
                            config = row.get("config") or {}
                            if theta is not None and (
                                _finite_float(config.get("theta")) is None
                                or abs(float(_finite_float(config.get("theta"))) - theta)
                                >= 1e-12
                            ):
                                continue
                            if window is not None and _integer(
                                config.get("recent_window")
                            ) != window:
                                continue
                            if alpha is not None and (
                                _finite_float(config.get("alpha")) is None
                                or abs(float(_finite_float(config.get("alpha"))) - alpha)
                                >= 1e-12
                            ):
                                continue
                            matching_rows.append(row)
                        for budget_type, budget_value in expected_budget_specs:
                            covered = budget_value is not None and any(
                                _descriptor_matches(
                                    descriptor,
                                    budget_type=budget_type,
                                    budget_value=budget_value,
                                )
                                for row in matching_rows
                                for descriptor in (
                                    (row.get("config") or {}).get(
                                        "budget_specifications"
                                    )
                                    or []
                                )
                            )
                            if not covered:
                                failures.append(
                                    f"{model}/{dataset}/{sample_id}/{method} is "
                                    "missing parameter/budget point "
                                    f"theta={theta}, window={window}, alpha={alpha}, "
                                    f"{budget_type}={budget_value}"
                                )

    if require_parity:
        parity = result.get("summary", {}).get("fullkv_parity", {})
        if parity.get("all_passed") is not True:
            mismatches = parity.get("mismatched_samples") or []
            detail = f": {mismatches}" if mismatches else ""
            failures.append(
                "FullKV/custom-cache generation parity did not pass" + detail
            )
        if expected_parity_records is not None:
            records = result.get("parity_records") or []
            matched = [
                row
                for row in records
                if row.get("token_match") is True and row.get("text_match") is True
            ]
            if len(matched) != int(expected_parity_records):
                failures.append(
                    "FullKV parity coverage mismatch: "
                    f"expected {int(expected_parity_records)}, observed {len(matched)}"
                )

    if require_fullkv_pairing:
        fullkv_keys = {
            (str(row.get("model")), str(row.get("dataset")), str(row.get("sample_id")))
            for row in successful
            if row.get("method") == "fullkv"
        }
        missing_pairs = sorted(
            {
                (str(row.get("model")), str(row.get("dataset")), str(row.get("sample_id")))
                for row in successful
                if row.get("method") != "fullkv"
            }
            - fullkv_keys
        )
        if missing_pairs:
            failures.append(
                f"compressed runs are missing paired FullKV rows: {missing_pairs[:10]}"
            )

    if parsed_min_parse_rate > 0.0:
        gsm8k_rows = [
            row for row in successful if "gsm8k" in str(row.get("dataset", "")).lower()
        ]
        if gsm8k_rows:
            parsed = sum(
                1
                for row in gsm8k_rows
                if (row.get("judgment") or {}).get("normalized_prediction")
                not in {None, ""}
            )
            parse_rate = parsed / len(gsm8k_rows)
            if parse_rate < parsed_min_parse_rate:
                failures.append(
                    f"GSM8K parse rate {parse_rate:.6f} is below "
                    f"{parsed_min_parse_rate:.6f}"
                )

    if require_cuda and not result.get("environment", {}).get("cuda", {}).get(
        "available", False
    ):
        failures.append("CUDA was required but the run did not record an available GPU")

    if require_driver and not result.get("environment", {}).get("cuda", {}).get(
        "driver_version"
    ):
        failures.append("NVIDIA driver provenance is missing")
    if require_clean_git and result.get("environment", {}).get("git", {}).get(
        "dirty"
    ) is not False:
        failures.append("paper results require a clean Git worktree")
    if require_model_revision:
        missing_resolved = [
            model
            for model, revision in (result.get("model_revisions") or {}).items()
            if not revision
        ]
        requested = result.get("requested_model_revisions") or {}
        missing_requested = [
            model for model in result.get("models", []) if not requested.get(model)
        ]
        if missing_requested:
            failures.append(
                f"requested model revision pins are missing for {missing_requested}"
            )
        if missing_resolved:
            failures.append(
                f"resolved model revisions are missing for {missing_resolved}"
            )
        mismatched = [
            model
            for model in result.get("models", [])
            if requested.get(model)
            and requested.get(model) != (result.get("model_revisions") or {}).get(model)
        ]
        if mismatched:
            failures.append(
                f"requested revisions did not resolve immutably for {mismatched}"
            )
    if require_frozen_manifest:
        dataset_names = {str(dataset.get("name")) for dataset in result.get("datasets", [])}
        frozen_names = set((result.get("dataset_manifest_hashes") or {}).keys())
        if dataset_names != frozen_names:
            failures.append("every paper dataset must use a hash-verified frozen manifest")
    if require_preflight:
        failed = [
            model
            for model, report in (result.get("model_preflights") or {}).items()
            if report.get("passed") is not True
        ]
        if failed or not result.get("model_preflights"):
            failures.append(f"target-model preflight did not pass for {failed}")

    if not result.get("experiment_fingerprint"):
        failures.append("experiment fingerprint is missing")
    if not result.get("environment", {}).get("git", {}).get("commit"):
        failures.append("Git commit provenance is missing")
    return failures


def qualification_report(
    result: dict[str, Any],
    *,
    require_parity: bool = True,
    require_cuda: bool = False,
    require_frozen_manifest: bool = False,
    require_model_revision: bool = False,
    require_driver: bool = False,
    require_clean_git: bool = False,
    require_preflight: bool = False,
    require_exact_niah: bool = False,
    require_no_truncation: bool = False,
    require_numerical_health: bool = True,
    require_generation_health: bool = True,
    require_fullkv_pairing: bool = False,
    expected_parity_records: int | None = None,
    min_gsm8k_parse_rate: float = 0.0,
) -> dict[str, Any]:
    failures = qualification_failures(
        result,
        require_parity=require_parity,
        require_cuda=require_cuda,
        require_frozen_manifest=require_frozen_manifest,
        require_model_revision=require_model_revision,
        require_driver=require_driver,
        require_clean_git=require_clean_git,
        require_preflight=require_preflight,
        require_exact_niah=require_exact_niah,
        require_no_truncation=require_no_truncation,
        require_numerical_health=require_numerical_health,
        require_generation_health=require_generation_health,
        require_fullkv_pairing=require_fullkv_pairing,
        expected_parity_records=expected_parity_records,
        min_gsm8k_parse_rate=min_gsm8k_parse_rate,
    )
    return {
        "passed": not failures,
        "require_parity": bool(require_parity),
        "require_cuda": bool(require_cuda),
        "require_frozen_manifest": bool(require_frozen_manifest),
        "require_model_revision": bool(require_model_revision),
        "require_driver": bool(require_driver),
        "require_clean_git": bool(require_clean_git),
        "require_preflight": bool(require_preflight),
        "require_exact_niah": bool(require_exact_niah),
        "require_no_truncation": bool(require_no_truncation),
        "require_numerical_health": bool(require_numerical_health),
        "require_generation_health": bool(require_generation_health),
        "require_fullkv_pairing": bool(require_fullkv_pairing),
        "expected_parity_records": expected_parity_records,
        "min_gsm8k_parse_rate": float(min_gsm8k_parse_rate),
        "failures": failures,
    }


def recompute_declared_qualification(result: dict[str, Any]) -> dict[str, Any]:
    """Re-run exactly the qualification contract stored in an artifact."""
    stored = ((result.get("summary") or {}).get("qualification"))
    if not isinstance(stored, dict):
        return {
            "passed": False,
            "stored_passed": False,
            "failures": ["stored qualification contract is missing"],
        }
    report = qualification_report(
        result,
        require_parity=bool(stored.get("require_parity", True)),
        require_cuda=bool(stored.get("require_cuda", False)),
        require_frozen_manifest=bool(stored.get("require_frozen_manifest", False)),
        require_model_revision=bool(stored.get("require_model_revision", False)),
        require_driver=bool(stored.get("require_driver", False)),
        require_clean_git=bool(stored.get("require_clean_git", False)),
        require_preflight=bool(stored.get("require_preflight", False)),
        require_exact_niah=bool(stored.get("require_exact_niah", False)),
        require_no_truncation=bool(stored.get("require_no_truncation", False)),
        require_numerical_health=bool(stored.get("require_numerical_health", True)),
        require_generation_health=bool(stored.get("require_generation_health", True)),
        require_fullkv_pairing=bool(stored.get("require_fullkv_pairing", False)),
        expected_parity_records=stored.get("expected_parity_records"),
        min_gsm8k_parse_rate=stored.get("min_gsm8k_parse_rate", 0.0),
    )
    report["stored_passed"] = stored.get("passed") is True
    if report["stored_passed"] is not True:
        report["failures"] = [
            *report.get("failures", []),
            "stored qualification did not pass",
        ]
        report["passed"] = False
    return report


__all__ = [
    "qualification_failures",
    "qualification_report",
    "recompute_declared_qualification",
]
