"""Aggregate raw experiment outputs into paper tables and figures."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import random
import math
from typing import Any, Iterable

from benchmarks.eval_metrics import (
    gsm8k_accuracy,
    hotpotqa_f1,
    niah_retrieval_match,
    token_f1,
)
from benchmarks.result_schema import assert_result_payload
from benchmarks.qualification import recompute_declared_qualification


METHOD_ORDER = ("fullkv", "streamingllm", "h2o", "snapkv", "chunkkv", "tdc_kv")


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
    gold = str(gold)
    family = _dataset_family(str(run.get("dataset", "")))
    if family == "gsm8k":
        return "gsm8k_accuracy", gsm8k_accuracy(prediction, gold)
    if family == "niah":
        return "niah_retrieval_accuracy", niah_retrieval_match(prediction, gold)
    if family == "hotpotqa":
        return "hotpotqa_f1", hotpotqa_f1(prediction, gold)
    return "token_f1", token_f1(prediction, gold)


def _requested_ratio(run: dict[str, Any]) -> float:
    if run.get("method") == "fullkv":
        return 1.0
    config = run.get("config") or {}
    for specification in config.get("budget_specifications") or []:
        if specification.get("type") == "ratio":
            return float(specification["value"])
    metrics = run.get("metrics") or {}
    return float(metrics.get("retention_ratio", 0.0))


def _variant(run: dict[str, Any]) -> str:
    config = run.get("config") or {}
    method = str(run.get("method", "unknown"))
    if method != "tdc_kv":
        return "baseline"
    explicit = config.get("experiment_variant")
    if explicit:
        return str(explicit)
    if config.get("tier1_score_mode") == "none":
        return "no_tier1"
    if config.get("protect_sink") is False:
        return "no_sink"
    if config.get("protect_recent") is False:
        return "no_recent"
    strategy = config.get("chunking_strategy", "sentence")
    if strategy == "token":
        return "token_level"
    if strategy == "fixed":
        return f"fixed_{config.get('fixed_chunk_size', 16)}"
    if config.get("attention_mode") == "all":
        return f"layers_{config.get('layer_weighting', 'linear')}"
    alpha = config.get("alpha")
    if alpha is not None and abs(float(alpha) - 0.6) > 1e-9:
        return f"alpha_{float(alpha):g}"
    return "default"


def load_paper_runs(
    paths: Iterable[str | Path],
    *,
    allow_unqualified: bool = False,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for path_value in paths:
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not allow_unqualified:
            assert_result_payload(payload, require_complete=True)
            qualification = recompute_declared_qualification(payload)
            if qualification.get("passed") is not True:
                raise ValueError(
                    f"Result is not paper-qualified: {path}: "
                    f"{qualification.get('failures')}"
                )
        seed = int(payload.get("environment", {}).get("seed", payload.get("grid", {}).get("seed", 0)))
        method_metadata = payload.get("method_metadata") or {}
        for index, raw in enumerate(payload.get("runs", [])):
            if raw.get("status") != "ok":
                continue
            identity = str(raw.get("run_key") or f"{path.resolve()}:{index}")
            previous = seen.get(identity)
            if previous is not None:
                if previous != raw:
                    raise ValueError(f"Conflicting duplicate run key: {identity}")
                continue
            seen[identity] = dict(raw)
            run = dict(raw)
            run["_seed"] = seed
            run["_source_file"] = str(path)
            run["_method_metadata"] = dict(
                method_metadata.get(str(raw.get("method")), {})
            )
            runs.append(run)
    return runs


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + (z * z) / total
    center = (proportion + (z * z) / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + (z * z) / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _execution_contract(run: dict[str, Any]) -> str:
    explicit = run.get("execution_contract_id")
    if explicit:
        return str(explicit)
    config = run.get("config") or {}
    fields = (
        run.get("model_revision"),
        run.get("protocol"),
        run.get("prompt_serialization"),
        config.get("dtype"),
        config.get("attention_mode"),
        config.get("layer_weighting"),
        config.get("prefill_block_size"),
        config.get("max_length"),
        config.get("max_new_tokens"),
        config.get("decode_policy"),
    )
    return json.dumps(fields, sort_keys=True, default=str)


def _method_config(run: dict[str, Any]) -> str:
    explicit = run.get("method_config_id")
    if explicit:
        return str(explicit)
    return json.dumps(run.get("config") or {}, sort_keys=True, default=str)


def _mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _std(values: Iterable[float | int | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return statistics.stdev(clean) if len(clean) > 1 else 0.0


def _bootstrap_mean_ci(
    values: Iterable[float | int | None],
    *,
    seed: int = 42,
    repetitions: int = 2000,
) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], clean[0]
    rng = random.Random(seed)
    means = sorted(
        sum(clean[rng.randrange(len(clean))] for _ in clean) / len(clean)
        for _ in range(repetitions)
    )
    return (
        means[int(0.025 * (len(means) - 1))],
        means[int(0.975 * (len(means) - 1))],
    )


def _runtime_value(run: dict[str, Any], field: str) -> float | None:
    runtime = run.get("runtime") or {}
    if field in runtime and runtime[field] is not None:
        return float(runtime[field])
    if field.endswith("_ms"):
        stage = field.removesuffix("_ms")
        value = (runtime.get("stages") or {}).get(stage, {}).get("elapsed_ms")
        return float(value) if value is not None else None
    return None


def aggregate_paper_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            str(run.get("model", "unknown")),
            str(run.get("dataset", "unknown")),
            str(run.get("method", "unknown")),
            _variant(run),
            _requested_ratio(run),
            _execution_contract(run),
            _method_config(run),
        )
        groups[key].append(run)

    rows: list[dict[str, Any]] = []
    for (
        model,
        dataset,
        method,
        variant,
        ratio,
        execution_contract_id,
        method_config_id,
    ), group_runs in groups.items():
        per_sample_scores: dict[str, list[float]] = defaultdict(list)
        observed_seeds: set[int] = set()
        metric_name = "unavailable"
        for run in group_runs:
            metric_name, score = _task_score(run)
            if score is not None:
                per_sample_scores[str(run.get("sample_id"))].append(float(score))
                observed_seeds.add(int(run.get("_seed", 0)))
        sample_scores = [
            sum(values) / len(values)
            for values in per_sample_scores.values()
            if values
        ]
        quality_ci_low, quality_ci_high = _bootstrap_mean_ci(sample_scores)
        wilson_low = wilson_high = None
        if _dataset_family(dataset) == "gsm8k":
            successes = sum(1 for score in sample_scores if score >= 0.5)
            wilson_low, wilson_high = wilson_interval(successes, len(sample_scores))
        metrics = [run.get("metrics") or {} for run in group_runs]
        structural = [run.get("structural_metrics") or {} for run in group_runs]
        cache_memory = [run.get("cache_memory") or {} for run in group_runs]
        metadata = next(
            (
                run.get("_method_metadata")
                for run in group_runs
                if run.get("_method_metadata")
            ),
            {},
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "dataset_family": _dataset_family(dataset),
                "method": method,
                "variant": variant,
                "implementation": metadata.get("implementation"),
                "reference_equivalence": metadata.get("reference_equivalence"),
                "paper_claim_level": metadata.get("paper_claim_level"),
                "execution_contract_id": execution_contract_id,
                "method_config_id": method_config_id,
                "retention_ratio": ratio,
                "compression_ratio": 1.0 - ratio,
                "compression_multiplier": 1.0 / ratio if ratio > 0 else None,
                "metric": metric_name,
                "quality_mean": _mean(sample_scores),
                "quality_std": _std(sample_scores),
                "quality_bootstrap_95ci_low": quality_ci_low,
                "quality_bootstrap_95ci_high": quality_ci_high,
                "quality_wilson_95ci_low": wilson_low,
                "quality_wilson_95ci_high": wilson_high,
                "seeds": len(observed_seeds),
                "samples": len(sample_scores),
                "repetitions": len(group_runs),
                "budget_utilization": _mean(
                    metric.get("budget_utilization") for metric in metrics
                ),
                "budget_overflow_max": max(
                    (int(metric.get("budget_overflow", 0)) for metric in metrics),
                    default=0,
                ),
                "generation_ms": _mean(
                    _runtime_value(run, "generation_ms") for run in group_runs
                ),
                "prefill_ms": _mean(
                    _runtime_value(run, "prefill_ms") for run in group_runs
                ),
                "scoring_ms": _mean(
                    _runtime_value(run, "scoring_ms") for run in group_runs
                ),
                "policy_ms": _mean(
                    _runtime_value(run, "policy_ms") for run in group_runs
                ),
                "decode_ms": _mean(
                    _runtime_value(run, "decode_ms") for run in group_runs
                ),
                "decode_tokens_per_second": _mean(
                    _runtime_value(run, "decode_tokens_per_second")
                    for run in group_runs
                ),
                "decode_ms_per_token": _mean(
                    _runtime_value(run, "decode_ms_per_token")
                    for run in group_runs
                ),
                "method_specific_ms": _mean(
                    _runtime_value(run, "method_specific_ms")
                    for run in group_runs
                ),
                "kv_gib_before": (
                    _mean(row.get("kv_bytes_before") for row in cache_memory) or 0.0
                )
                / (1024.0**3),
                "kv_gib_after": (
                    _mean(row.get("kv_bytes_after") for row in cache_memory) or 0.0
                )
                / (1024.0**3),
                "kv_gib_saved": (
                    _mean(row.get("kv_bytes_saved") for row in cache_memory) or 0.0
                )
                / (1024.0**3),
                "peak_vram_gib": (
                    _mean(
                        _runtime_value(run, "max_peak_allocated_bytes")
                        for run in group_runs
                    )
                    or 0.0
                )
                / (1024.0**3),
                "evidence_token_retention": _mean(
                    row.get("evidence_token_retention") for row in structural
                ),
                "evidence_token_eviction_ratio": _mean(
                    row.get(
                        "evidence_token_eviction_ratio",
                        row.get("global_eviction_ratio"),
                    )
                    for row in structural
                ),
                "evidence_chunk_survival": _mean(
                    row.get("evidence_chunk_survival") for row in structural
                ),
                "head_consensus": _mean(
                    row.get("head_consensus") for row in structural
                ),
                "evidence_depth": _mean(
                    row.get("evidence_depth_min") for row in structural
                ),
                "tier0_chunks": _mean(
                    (run.get("tier_counts") or {}).get("tier0_chunks")
                    for run in group_runs
                ),
                "tier1_chunks": _mean(
                    (run.get("tier_counts") or {}).get("tier1_chunks")
                    for run in group_runs
                ),
                "tier2_chunks": _mean(
                    (run.get("tier_counts") or {}).get("tier2_chunks")
                    for run in group_runs
                ),
            }
        )
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
        quality = row.get("quality_mean")
        row["normalized_quality_ratio"] = (
            100.0 * float(quality) / float(reference)
            if quality is not None and reference not in {None, 0.0}
            else None
        )

    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["dataset"],
            method_rank.get(row["method"], 99),
            row["variant"],
            -row["retention_ratio"],
        ),
    )


def paired_significance_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired tests over unique samples, collapsing deterministic repetitions."""
    repeated: dict[
        tuple[str, str, float, str, str, str, str], list[float]
    ] = defaultdict(list)
    for run in runs:
        if run.get("method") == "tdc_kv" and _variant(run) != "default":
            continue
        _, score = _task_score(run)
        if score is None:
            continue
        key = (
            str(run.get("model")),
            str(run.get("dataset")),
            _requested_ratio(run),
            _execution_contract(run),
            str(run.get("sample_id")),
            str(run.get("method")),
            _method_config(run),
        )
        repeated[key].append(float(score))
    scored = {
        key: sum(values) / len(values) for key, values in repeated.items() if values
    }

    comparisons: list[dict[str, Any]] = []
    base_dimensions = sorted(
        {
            (model, dataset, ratio, contract, method_config)
            for model, dataset, ratio, contract, _sample, method, method_config in scored
            if method == "tdc_kv"
        }
    )
    for model, dataset, ratio, contract, tdc_method_config in base_dimensions:
        for baseline in METHOD_ORDER:
            if baseline in {"tdc_kv", "fullkv"}:
                continue
            baseline_configs = sorted(
                {
                    method_config
                    for (
                        key_model,
                        key_dataset,
                        key_ratio,
                        key_contract,
                        _sample,
                        method,
                        method_config,
                    ) in scored
                    if method == baseline
                    and key_model == model
                    and key_dataset == dataset
                    and key_ratio == ratio
                    and key_contract == contract
                }
            )
            for baseline_method_config in baseline_configs:
                pairs: list[tuple[float, float]] = []
                for key, tdc_score in scored.items():
                    (
                        key_model,
                        key_dataset,
                        key_ratio,
                        key_contract,
                        sample,
                        method,
                        key_method_config,
                    ) = key
                    if (
                        method != "tdc_kv"
                        or key_method_config != tdc_method_config
                        or key_model != model
                        or key_dataset != dataset
                        or key_ratio != ratio
                        or key_contract != contract
                    ):
                        continue
                    baseline_score = scored.get(
                        (
                            model,
                            dataset,
                            ratio,
                            contract,
                            sample,
                            baseline,
                            baseline_method_config,
                        )
                    )
                    if baseline_score is not None:
                        pairs.append((tdc_score, baseline_score))
                if not pairs:
                    continue
                deltas = [tdc - base for tdc, base in pairs]
                delta_mean = _mean(deltas) or 0.0
                delta_std = _std(deltas)
                rng = random.Random(42)
                bootstrap = []
                for _ in range(2000):
                    sampled = [deltas[rng.randrange(len(deltas))] for _ in deltas]
                    bootstrap.append(sum(sampled) / len(sampled))
                bootstrap.sort()
                lower = bootstrap[int(0.025 * (len(bootstrap) - 1))]
                upper = bootstrap[int(0.975 * (len(bootstrap) - 1))]
                wilcoxon_p = None
                paired_t_p = None
                try:
                    from scipy.stats import ttest_rel, wilcoxon

                    if any(abs(delta) > 0 for delta in deltas):
                        wilcoxon_p = float(
                            wilcoxon(
                                [pair[0] for pair in pairs],
                                [pair[1] for pair in pairs],
                            ).pvalue
                        )
                    else:
                        wilcoxon_p = 1.0
                    if len(pairs) > 1 and delta_std > 0:
                        paired_t_p = float(
                            ttest_rel(
                                [pair[0] for pair in pairs],
                                [pair[1] for pair in pairs],
                            ).pvalue
                        )
                except (ImportError, ValueError, ZeroDivisionError):
                    pass
                comparisons.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "retention_ratio": ratio,
                        "execution_contract_id": contract,
                        "tdc_method_config_id": tdc_method_config,
                        "baseline": baseline,
                        "baseline_method_config_id": baseline_method_config,
                        "pairs": len(pairs),
                        "tdc_mean": _mean(pair[0] for pair in pairs),
                        "baseline_mean": _mean(pair[1] for pair in pairs),
                        "mean_delta": delta_mean,
                        "delta_std": delta_std,
                        "cohens_dz": delta_mean / delta_std if delta_std > 0 else None,
                        "bootstrap_95ci_low": lower,
                        "bootstrap_95ci_high": upper,
                        "wilcoxon_p": wilcoxon_p,
                        "paired_t_p": paired_t_p,
                    }
                )
    for source, destination in (
        ("wilcoxon_p", "wilcoxon_p_holm"),
        ("paired_t_p", "paired_t_p_holm"),
    ):
        indexed = sorted(
            (
                (index, float(row[source]))
                for index, row in enumerate(comparisons)
                if row.get(source) is not None
            ),
            key=lambda item: item[1],
        )
        running = 0.0
        total = len(indexed)
        for rank, (index, p_value) in enumerate(indexed):
            adjusted = min(1.0, p_value * (total - rank))
            running = max(running, adjusted)
            comparisons[index][destination] = running
        for row in comparisons:
            row.setdefault(destination, None)
    return comparisons


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _format(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def write_markdown_tables(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    headline = [
        row
        for row in rows
        if row["method"] == "fullkv"
        or abs(row["retention_ratio"] - 0.20) < 1e-9
    ]
    lines = [
        "# Main Results at 20% KV Retention",
        "",
        "| Model | Dataset | Method | Fidelity | Variant | Quality (mean ± std) | Tokens/s | Peak VRAM GiB | Evidence eviction |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in headline:
        quality = f"{_format(row['quality_mean'])} ± {_format(row['quality_std'])}"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['method']} | "
            f"{row.get('reference_equivalence') or 'unknown'} | {row['variant']} | {quality} | "
            f"{_format(row['decode_tokens_per_second'], 2)} | {_format(row['peak_vram_gib'], 2)} | "
            f"{_format(row['evidence_token_eviction_ratio'])} |"
        )
    (output / "table1_main_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    curves = [
        "# Matched-Budget Quality Curves",
        "",
        "| Model | Dataset | Method | Variant | Retention | Metric | Quality (mean ± std) |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in rows:
        curves.append(
            f"| {row['model']} | {row['dataset']} | {row['method']} | {row['variant']} | "
            f"{100 * row['retention_ratio']:.2f}% | {row['metric']} | "
            f"{_format(row['quality_mean'])} ± {_format(row['quality_std'])} |"
        )
    (output / "table2_quality_curves.md").write_text(
        "\n".join(curves) + "\n", encoding="utf-8"
    )

    efficiency = [
        "# Efficiency Results",
        "",
        "| Model | Dataset | Method | Variant | Retention | Prefill ms | Scoring ms | Policy ms | Decode ms/token | Tokens/s | KV GiB after | Peak VRAM GiB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        efficiency.append(
            f"| {row['model']} | {row['dataset']} | {row['method']} | {row['variant']} | "
            f"{100 * row['retention_ratio']:.2f}% | {_format(row['prefill_ms'], 2)} | "
            f"{_format(row['scoring_ms'], 2)} | {_format(row['policy_ms'], 2)} | "
            f"{_format(row['decode_ms_per_token'], 2)} | {_format(row['decode_tokens_per_second'], 2)} | "
            f"{_format(row['kv_gib_after'], 3)} | "
            f"{_format(row['peak_vram_gib'], 2)} |"
        )
    (output / "table3_efficiency.md").write_text(
        "\n".join(efficiency) + "\n", encoding="utf-8"
    )


def generate_figures(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(METHOD_ORDER, ("#777777", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2")))

    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for model in sorted({row["model"] for row in dataset_rows}):
            subset = [row for row in dataset_rows if row["model"] == model]
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            series = sorted(
                {(row["method"], row["variant"]) for row in subset},
                key=lambda item: (METHOD_ORDER.index(item[0]) if item[0] in METHOD_ORDER else 99, item[1]),
            )
            for method, variant in series:
                points = sorted(
                    (
                        row
                        for row in subset
                        if row["method"] == method and row["variant"] == variant
                    ),
                    key=lambda row: row["retention_ratio"],
                )
                points = [row for row in points if row["quality_mean"] is not None]
                if not points:
                    continue
                ax.errorbar(
                    [100 * row["retention_ratio"] for row in points],
                    [row["quality_mean"] for row in points],
                    yerr=[row["quality_std"] for row in points],
                    marker="o",
                    label=method if variant in {"default", "baseline"} else f"{method}:{variant}",
                    color=colors.get(method),
                )
            ax.set_xlabel("KV retention (%)")
            ax.set_ylabel("Task score")
            ax.set_title(f"{dataset} — {model}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            safe = f"{dataset}_{model}".replace("/", "_").replace(" ", "_")
            fig.tight_layout()
            fig.savefig(output / f"quality_curve_{safe}.png", dpi=300)
            plt.close(fig)

    evidence_rows = [
        row for row in rows if row["evidence_token_eviction_ratio"] is not None
    ]
    if evidence_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        series = sorted({(row["method"], row["variant"]) for row in evidence_rows})
        for method, variant in series:
            points = sorted(
                (
                    row
                    for row in evidence_rows
                    if row["method"] == method and row["variant"] == variant
                ),
                key=lambda row: row["retention_ratio"],
            )
            if points:
                ax.plot(
                    [100 * row["retention_ratio"] for row in points],
                    [row["evidence_token_eviction_ratio"] for row in points],
                    marker="o",
                    label=method if variant in {"default", "baseline"} else f"{method}:{variant}",
                    color=colors.get(method),
                )
        ax.set_xlabel("KV retention (%)")
        ax.set_ylabel("Evidence-token eviction ratio (lower is better)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "evidence_eviction_vs_retention.png", dpi=300)
        plt.close(fig)

    efficiency_rows = [
        row
        for row in rows
        if row["method"] == "fullkv"
        or abs(row["retention_ratio"] - 0.20) < 1e-9
    ]
    if efficiency_rows:
        labels = [f"{row['method']}\n{row['dataset']}" for row in efficiency_rows]
        fig, axes = plt.subplots(2, 1, figsize=(max(8, len(labels) * 0.45), 7))
        axes[0].bar(range(len(labels)), [row["decode_tokens_per_second"] or 0 for row in efficiency_rows])
        axes[0].set_ylabel("Decode tokens/s")
        axes[1].bar(range(len(labels)), [row["peak_vram_gib"] or 0 for row in efficiency_rows])
        axes[1].set_ylabel("Peak VRAM (GiB)")
        axes[1].set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "efficiency_25pct.png", dpi=300)
        plt.close(fig)

    niah_rows = [
        row
        for row in rows
        if row["dataset_family"] == "niah"
        and row["quality_mean"] is not None
        and row["evidence_depth"] is not None
    ]
    for model in sorted({row["model"] for row in niah_rows}):
        for method in sorted({row["method"] for row in niah_rows}):
            subset = [
                row
                for row in niah_rows
                if row["model"] == model
                and row["method"] == method
                and row["variant"] in {"default", "baseline"}
            ]
            depths = sorted({round(float(row["evidence_depth"]), 3) for row in subset})
            ratios = sorted({float(row["retention_ratio"]) for row in subset})
            if len(depths) < 2 or not ratios:
                continue
            matrix = [[float("nan") for _ in ratios] for _ in depths]
            for row in subset:
                depth = round(float(row["evidence_depth"]), 3)
                matrix[depths.index(depth)][ratios.index(float(row["retention_ratio"]))] = float(row["quality_mean"])
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(ratios)), [f"{100 * ratio:g}%" for ratio in ratios])
            ax.set_yticks(range(len(depths)), [f"{100 * depth:.1f}%" for depth in depths])
            ax.set_xlabel("KV retention")
            ax.set_ylabel("Observed needle depth")
            ax.set_title(f"NIAH — {method} — {model}")
            fig.colorbar(image, ax=ax, label="Retrieval accuracy")
            safe = f"{method}_{model}".replace("/", "_").replace(" ", "_")
            fig.tight_layout()
            fig.savefig(output / f"niah_heatmap_{safe}.png", dpi=300)
            plt.close(fig)

    tier_rows = [
        row
        for row in rows
        if row["method"] == "tdc_kv"
        and row["tier0_chunks"] is not None
        and abs(row["retention_ratio"] - 0.20) < 1e-9
    ]
    if tier_rows:
        labels = [f"{row['dataset']}\n{row['variant']}" for row in tier_rows]
        tier0 = [row["tier0_chunks"] or 0 for row in tier_rows]
        tier1 = [row["tier1_chunks"] or 0 for row in tier_rows]
        tier2 = [row["tier2_chunks"] or 0 for row in tier_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.8))
        x = list(range(len(labels)))
        ax.bar(x, tier0, label="Tier 0")
        ax.bar(x, tier1, bottom=tier0, label="Tier 1")
        ax.bar(x, tier2, bottom=[a + b for a, b in zip(tier0, tier1)], label="Tier 2")
        ax.set_ylabel("Average chunks")
        ax.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "tier_distribution_25pct.png", dpi=300)
        plt.close(fig)

    ablation_rows = [
        row
        for row in rows
        if row["method"] == "tdc_kv"
        and row["variant"] != "default"
        and abs(row["retention_ratio"] - 0.20) < 1e-9
        and row["quality_mean"] is not None
    ]
    if ablation_rows:
        labels = [f"{row['variant']}\n{row['dataset']}" for row in ablation_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.8))
        ax.bar(range(len(labels)), [row["quality_mean"] for row in ablation_rows])
        ax.set_ylabel("Task score")
        ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "ablation_quality_25pct.png", dpi=300)
        plt.close(fig)


__all__ = [
    "aggregate_paper_rows",
    "generate_figures",
    "load_paper_runs",
    "paired_significance_rows",
    "wilson_interval",
    "write_csv",
    "write_markdown_tables",
]
