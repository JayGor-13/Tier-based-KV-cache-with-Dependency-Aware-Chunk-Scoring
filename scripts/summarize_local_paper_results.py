"""Summarize local TDC-KV paper-sweep JSON files into table artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import glob
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import (
    gsm8k_accuracy,
    hotpotqa_f1,
    niah_retrieval_match,
    token_f1,
)
from benchmarks.paper_reporting import load_paper_runs
from benchmarks.paper_reporting import wilson_interval
from benchmarks.io_utils import write_json_atomic


CONFIG_FIELDS = (
    "theta",
    "alpha",
    "beta",
    "recent_window",
    "chunking_strategy",
    "fixed_chunk_size",
    "min_chunk_tokens",
    "max_chunk_tokens",
    "dependency_top_k",
    "tier1_score_mode",
    "attention_mode",
    "layer_weighting",
    "protect_sink",
    "protect_recent",
    "allow_level2_fallback",
    "decode_policy",
    "experiment_variant",
    "dtype",
    "attn_implementation",
    "max_length",
    "max_new_tokens",
    "prefill_block_size",
    "seed",
)


def _dataset_family(name: str) -> str:
    normalized = str(name).lower()
    if "gsm8k" in normalized:
        return "gsm8k"
    if "niah" in normalized or "needle" in normalized:
        return "niah"
    if "hotpot" in normalized:
        return "hotpotqa"
    return "generic"


def _task_score(run: dict[str, Any]) -> tuple[str, float | None]:
    prediction = str(run.get("evicted_prediction", run.get("prediction", "")))
    gold = run.get("gold")
    if gold is None:
        return "unavailable", None
    family = _dataset_family(str(run.get("dataset", "")))
    if family == "gsm8k":
        return "gsm8k_accuracy", gsm8k_accuracy(prediction, str(gold))
    if family == "niah":
        return "niah_retrieval_accuracy", niah_retrieval_match(prediction, str(gold))
    if family == "hotpotqa":
        return "hotpotqa_f1", hotpotqa_f1(prediction, str(gold))
    return "token_f1", token_f1(prediction, str(gold))


def _requested_retention_ratio(run: dict[str, Any]) -> float:
    if run.get("method") == "fullkv":
        return 1.0
    config = run.get("config") or {}
    for specification in config.get("budget_specifications") or []:
        if specification.get("type") == "ratio":
            return float(specification["value"])
    metrics = run.get("metrics") or {}
    if metrics.get("retention_ratio") is not None:
        return float(metrics["retention_ratio"])
    return 1.0


def _mean(values: Iterable[Any]) -> float | None:
    clean = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            clean.append(number)
    return sum(clean) / len(clean) if clean else None


def _runtime_value(run: dict[str, Any], field: str) -> float | None:
    runtime = run.get("runtime") or {}
    if field in runtime and runtime[field] is not None:
        return float(runtime[field])
    if field.endswith("_ms"):
        stage = field.removesuffix("_ms")
        value = (runtime.get("stages") or {}).get(stage, {}).get("elapsed_ms")
        return float(value) if value is not None else None
    return None


def _group_key(run: dict[str, Any]) -> tuple[Any, ...]:
    config = run.get("config") or {}
    return (
        str(run.get("model", "unknown")),
        str(run.get("dataset", "unknown")),
        str(run.get("method", "unknown")),
        _requested_retention_ratio(run),
        str(run.get("execution_contract_id") or "legacy"),
        str(run.get("method_config_id") or "legacy"),
        str(run.get("model_revision") or "unknown"),
        str(run.get("protocol") or config.get("protocol") or "default"),
        str(run.get("prompt_serialization") or config.get("prompt_serialization") or "unknown"),
        *(config.get(field) for field in CONFIG_FIELDS),
    )


def aggregate_algorithm_parameter_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for index, run in enumerate(runs):
        run_key = str(run.get("run_key") or f"legacy:{index}")
        previous = seen.get(run_key)
        if previous is not None:
            if previous != run:
                raise ValueError(f"Conflicting duplicate run key: {run_key}")
            continue
        seen[run_key] = run
        deduplicated.append(run)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in deduplicated:
        grouped[_group_key(run)].append(run)

    rows: list[dict[str, Any]] = []
    for key, group_runs in grouped.items():
        (
            model,
            dataset,
            method,
            requested_ratio,
            execution_contract_id,
            method_config_id,
            model_revision,
            protocol,
            prompt_serialization,
            *config_values,
        ) = key
        config = dict(zip(CONFIG_FIELDS, config_values))
        scored = [_task_score(run) for run in group_runs]
        metric = next((name for name, score in scored if score is not None), scored[0][0])
        scores = [score for _, score in scored]
        metrics = [run.get("metrics") or {} for run in group_runs]
        cache_memory = [run.get("cache_memory") or {} for run in group_runs]
        structural = [run.get("structural_metrics") or {} for run in group_runs]
        tier_counts = [run.get("tier_counts") or {} for run in group_runs]
        metadata = next(
            (
                run.get("_method_metadata")
                for run in group_runs
                if run.get("_method_metadata")
            ),
            {},
        )
        requested_ratio = float(requested_ratio)
        quality_mean = _mean(scores)
        wilson_low = wilson_high = None
        if _dataset_family(str(dataset)) == "gsm8k":
            binary_scores = [float(score) for score in scores if score is not None]
            wilson_low, wilson_high = wilson_interval(
                sum(1 for score in binary_scores if score >= 0.5),
                len(binary_scores),
            )
        row = {
            "model": model,
            "dataset": dataset,
            "dataset_family": _dataset_family(str(dataset)),
            "method": method,
            "implementation": metadata.get("implementation"),
            "reference_equivalence": metadata.get("reference_equivalence"),
            "paper_claim_level": metadata.get("paper_claim_level"),
            "execution_contract_id": execution_contract_id,
            "method_config_id": method_config_id,
            "model_revision": model_revision,
            "protocol": protocol,
            "prompt_serialization": prompt_serialization,
            "requested_retention_ratio": requested_ratio,
            "requested_compression_ratio": 1.0 - requested_ratio,
            "requested_compression_multiplier": (
                1.0 / requested_ratio if requested_ratio > 0.0 else None
            ),
            **config,
            "samples": len({str(run.get("sample_id")) for run in group_runs}),
            "runs": len(group_runs),
            "metric": metric,
            "quality_mean": quality_mean,
            "quality_wilson_95ci_low": wilson_low,
            "quality_wilson_95ci_high": wilson_high,
            "actual_retention_ratio": _mean(
                metric_row.get("retention_ratio") for metric_row in metrics
            ),
            "actual_compression_ratio": _mean(
                metric_row.get("compression_ratio") for metric_row in metrics
            ),
            "actual_compression_multiplier": _mean(
                metric_row.get("compression_multiplier") for metric_row in metrics
            ),
            "budget_utilization": _mean(
                metric_row.get("budget_utilization") for metric_row in metrics
            ),
            "budget_shortfall_max": max(
                (int(metric_row.get("budget_shortfall", 0) or 0) for metric_row in metrics),
                default=0,
            ),
            "budget_overflow_max": max(
                (int(metric_row.get("budget_overflow", 0) or 0) for metric_row in metrics),
                default=0,
            ),
            "sequence_length": _mean(run.get("sequence_length") for run in group_runs),
            "kept_tokens": _mean(run.get("kept_tokens") for run in group_runs),
            "removed_tokens": _mean(run.get("removed_tokens") for run in group_runs),
            "num_chunks": _mean(run.get("num_chunks") for run in group_runs),
            "avg_chunk_size": _mean(run.get("avg_chunk_size") for run in group_runs),
            "kv_gib_before": (_mean(row.get("kv_bytes_before") for row in cache_memory) or 0.0)
            / (1024.0**3),
            "kv_gib_after": (_mean(row.get("kv_bytes_after") for row in cache_memory) or 0.0)
            / (1024.0**3),
            "kv_gib_saved": (_mean(row.get("kv_bytes_saved") for row in cache_memory) or 0.0)
            / (1024.0**3),
            "prefill_ms": _mean(_runtime_value(run, "prefill_ms") for run in group_runs),
            "scoring_ms": _mean(_runtime_value(run, "scoring_ms") for run in group_runs),
            "policy_ms": _mean(_runtime_value(run, "policy_ms") for run in group_runs),
            "decode_ms": _mean(_runtime_value(run, "decode_ms") for run in group_runs),
            "method_specific_ms": _mean(
                _runtime_value(run, "method_specific_ms") for run in group_runs
            ),
            "decode_tokens_per_second": _mean(
                _runtime_value(run, "decode_tokens_per_second") for run in group_runs
            ),
            "decode_ms_per_token": _mean(
                _runtime_value(run, "decode_ms_per_token") for run in group_runs
            ),
            "evidence_token_retention": _mean(
                row.get("evidence_token_retention") for row in structural
            ),
            "evidence_chunk_survival": _mean(
                row.get("evidence_chunk_survival") for row in structural
            ),
            "tier0_chunks": _mean(row.get("tier0") for row in tier_counts),
            "tier1_chunks": _mean(row.get("tier1") for row in tier_counts),
            "tier2_chunks": _mean(row.get("tier2") for row in tier_counts),
        }
        rows.append(row)

    fullkv_quality = {
        (row["model"], row["dataset"], row["execution_contract_id"]): row[
            "quality_mean"
        ]
        for row in rows
        if row["method"] == "fullkv" and row["quality_mean"] is not None
    }
    for row in rows:
        reference = fullkv_quality.get(
            (row["model"], row["dataset"], row["execution_contract_id"])
        )
        row["normalized_quality_ratio"] = (
            100.0 * float(row["quality_mean"]) / float(reference)
            if row["quality_mean"] is not None and reference not in {None, 0.0}
            else None
        )

    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["dataset"],
            row["method"],
            -float(row["requested_retention_ratio"]),
            str(row.get("theta")),
            str(row.get("recent_window")),
            str(row.get("alpha")),
        ),
    )


def _discover_inputs(patterns: list[str], *, allow_glob: bool = False) -> list[Path]:
    paths: list[str] = []
    for pattern in patterns:
        if glob.has_magic(pattern) and not allow_glob:
            raise ValueError(
                "Wildcard discovery is disabled; use the suite manifest's explicit outputs."
            )
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    selected = []
    for path in dict.fromkeys(paths):
        if path.endswith((".checkpoint.json", ".checkpoint.sqlite")) or path.endswith(
            "suite_manifest.json"
        ):
            continue
        candidate = Path(path)
        if candidate.exists():
            selected.append(candidate)
    if not selected:
        raise FileNotFoundError("No result JSON files matched --inputs.")
    return selected


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headline = [
        row
        for row in rows
        if row["method"] == "fullkv"
        or row["requested_retention_ratio"] in {0.5, 0.3, 0.2, 0.1}
    ]
    lines = [
        "# Local Paper Sweep Summary",
        "",
        "| Model | Dataset | Method | Fidelity | Retention | Compression | Theta | Alpha | Recent | Metric | Quality | KV saved GiB | Tokens/s |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in headline:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['method']} | "
            f"{row.get('reference_equivalence') or 'unknown'} | "
            f"{100.0 * float(row['requested_retention_ratio']):.1f}% | "
            f"{100.0 * float(row['requested_compression_ratio']):.1f}% | "
            f"{_format(row.get('theta'))} | {_format(row.get('alpha'))} | "
            f"{_format(row.get('recent_window'), 0)} | {row['metric']} | "
            f"{_format(row['quality_mean'])} | {_format(row['kv_gib_saved'], 4)} | "
            f"{_format(row['decode_tokens_per_second'], 2)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/local_paper/artifacts")
    parser.add_argument("--allow-glob", action="store_true")
    args = parser.parse_args()

    paths = _discover_inputs(args.inputs, allow_glob=args.allow_glob)
    runs = load_paper_runs(paths)
    rows = aggregate_algorithm_parameter_rows(runs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "algorithm_parameter_grid.csv")
    write_json_atomic(output / "algorithm_parameter_grid.json", rows)
    write_markdown(rows, output / "algorithm_parameter_grid.md")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "successful_runs_loaded": len(runs),
        "aggregate_rows": len(rows),
        "outputs": {
            "csv": str((output / "algorithm_parameter_grid.csv").resolve()),
            "json": str((output / "algorithm_parameter_grid.json").resolve()),
            "markdown": str((output / "algorithm_parameter_grid.md").resolve()),
        },
    }
    write_json_atomic(output / "artifact_manifest.json", manifest)
    print(f"Loaded {len(runs)} successful runs from {len(paths)} files.")
    print(f"Wrote {len(rows)} aggregate rows to {output}.")


if __name__ == "__main__":
    main()
