"""Hard qualification checks for paper-grade experiment outputs."""

from __future__ import annotations

from typing import Any


def qualification_failures(
    result: dict[str, Any],
    *,
    require_parity: bool = True,
    require_cuda: bool = False,
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

    for index, row in enumerate(successful):
        identity = (
            f"run[{index}] {row.get('model')}/{row.get('dataset')}/"
            f"{row.get('sample_id')}/{row.get('method')}"
        )
        if not row.get("run_key"):
            failures.append(f"{identity} is missing its deterministic run key")
        if max_new_tokens > 0 and not row.get("generated_token_ids"):
            failures.append(f"{identity} generated no tokens")

        runtime = row.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("stages"):
            failures.append(f"{identity} has no stage runtime measurements")

        if row.get("method") == "fullkv":
            continue
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

    if require_parity:
        parity = result.get("summary", {}).get("fullkv_parity", {})
        if parity.get("all_passed") is not True:
            failures.append("FullKV/custom-cache generation parity did not pass")

    if require_cuda and not result.get("environment", {}).get("cuda", {}).get(
        "available", False
    ):
        failures.append("CUDA was required but the run did not record an available GPU")

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
) -> dict[str, Any]:
    failures = qualification_failures(
        result,
        require_parity=require_parity,
        require_cuda=require_cuda,
    )
    return {
        "passed": not failures,
        "require_parity": bool(require_parity),
        "require_cuda": bool(require_cuda),
        "failures": failures,
    }


__all__ = ["qualification_failures", "qualification_report"]
