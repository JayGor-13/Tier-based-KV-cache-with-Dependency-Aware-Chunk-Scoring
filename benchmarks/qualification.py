"""Hard qualification checks for paper-grade experiment outputs."""

from __future__ import annotations

import math
from typing import Any


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

    grid = result.get("grid", {})
    max_new_tokens = int(grid.get("max_new_tokens", 0) or 0)
    min_utilization = float(grid.get("min_budget_utilization", 0.99))
    max_shortfall = int(grid.get("max_budget_shortfall_tokens", 1))
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
        if require_no_truncation and row.get("prompt_truncated") is not False:
            failures.append(f"{identity} used a truncated prompt")

        runtime = row.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("stages"):
            failures.append(f"{identity} has no stage runtime measurements")
        elif any(
            not math.isfinite(float(stage.get("elapsed_ms", float("nan"))))
            for stage in runtime.get("stages", {}).values()
            if isinstance(stage, dict)
        ):
            failures.append(f"{identity} has a non-finite runtime measurement")

        cache_memory = row.get("cache_memory") or {}
        for field in ("kv_bytes_before", "kv_bytes_after", "kv_bytes_saved"):
            if field not in cache_memory:
                failures.append(f"{identity} is missing cache-memory field `{field}`")

        if row.get("method") == "fullkv":
            metrics = row.get("metrics") or {}
        else:
            metrics = row.get("metrics") or {}
            overflow = int(metrics.get("budget_overflow", 0) or 0)
            shortfall = int(metrics.get("budget_shortfall", 0) or 0)
            utilization = float(metrics.get("budget_utilization", 0.0) or 0.0)
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
            if int(structural.get("critical_token_count", 0) or 0) <= 0:
                failures.append(
                    f"{identity} localized no answer-critical evidence tokens"
                )
            localization = structural.get("evidence_localization_rate")
            if localization is None or float(localization) < 1.0:
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
    requested_methods = [str(method) for method in grid.get("methods", [])]
    for model in result.get("models", []):
        for dataset, samples in sample_manifest.items():
            for sample in samples:
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
                    for ratio in grid.get("budget_ratios", []):
                        if not any(
                            descriptor.get("type") == "ratio"
                            and abs(float(descriptor.get("value")) - float(ratio)) < 1e-12
                            for descriptor in descriptors
                        ):
                            failures.append(
                                f"{model}/{dataset}/{sample_id}/{method} is missing "
                                f"budget ratio {ratio}"
                            )
                    for budget in grid.get("budgets", []):
                        if not any(
                            descriptor.get("type") == "absolute"
                            and int(descriptor.get("value")) == int(budget)
                            for descriptor in descriptors
                        ):
                            failures.append(
                                f"{model}/{dataset}/{sample_id}/{method} is missing "
                                f"absolute budget {budget}"
                            )
                    if method == "tdc_kv":
                        parameter_points = [
                            (float(theta), int(window), float(alpha))
                            for theta in grid.get("thetas", [])
                            for window in grid.get("recent_windows", [])
                            for alpha in grid.get("alphas", [])
                        ]
                    elif method == "streamingllm":
                        parameter_points = [(None, None, None)]
                    else:
                        parameter_points = [
                            (None, int(window), None)
                            for window in grid.get("recent_windows", [])
                        ]
                    expected_budget_specs = [
                        ("ratio", float(value))
                        for value in grid.get("budget_ratios", [])
                    ] + [
                        ("absolute", int(value)) for value in grid.get("budgets", [])
                    ]
                    for theta, window, alpha in parameter_points:
                        matching_rows = []
                        for row in method_rows:
                            config = row.get("config") or {}
                            if theta is not None and abs(
                                float(config.get("theta")) - theta
                            ) >= 1e-12:
                                continue
                            if window is not None and int(
                                config.get("recent_window")
                            ) != window:
                                continue
                            if alpha is not None and abs(
                                float(config.get("alpha")) - alpha
                            ) >= 1e-12:
                                continue
                            matching_rows.append(row)
                        for budget_type, budget_value in expected_budget_specs:
                            covered = any(
                                descriptor.get("type") == budget_type
                                and (
                                    abs(float(descriptor.get("value")) - budget_value)
                                    < 1e-12
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
            failures.append("FullKV/custom-cache generation parity did not pass")

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
        "failures": failures,
    }


__all__ = ["qualification_failures", "qualification_report"]
