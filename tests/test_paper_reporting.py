from benchmarks.paper_reporting import (
    aggregate_paper_rows,
    paired_significance_rows,
    wilson_interval,
)


def _run(method, score, sample, seed):
    return {
        "status": "ok",
        "method": method,
        "model": "model",
        "dataset": "niah_8k_d50",
        "sample_id": sample,
        "_seed": seed,
        "config": {
            "budget_specifications": [{"type": "ratio", "value": 0.25}],
            "alpha": 0.6 if method == "tdc_kv" else None,
        },
        "evicted_prediction": "KEY" if score else "wrong",
        "gold": "KEY",
        "metrics": {"budget_utilization": 1.0, "budget_overflow": 0},
        "runtime": {"decode_tokens_per_second": 10.0},
        "structural_metrics": {"global_eviction_ratio": 0.0},
    }


def test_paper_rows_aggregate_seeds_and_paired_samples():
    runs = []
    for seed in (13, 42):
        runs.extend(
            [
                _run("tdc_kv", 1, "one", seed),
                _run("h2o", 0, "one", seed),
            ]
        )
    rows = aggregate_paper_rows(runs)
    tdc = next(row for row in rows if row["method"] == "tdc_kv")
    comparisons = paired_significance_rows(runs)
    h2o = next(row for row in comparisons if row["baseline"] == "h2o")
    assert tdc["quality_mean"] == 1.0
    assert tdc["seeds"] == 2
    assert tdc["samples"] == 1
    assert tdc["repetitions"] == 2
    assert h2o["pairs"] == 1
    assert h2o["mean_delta"] == 1.0
    assert h2o["wilcoxon_p_holm"] is not None


def test_explicit_default_variant_keeps_tuned_alpha_in_main_statistics():
    tdc = _run("tdc_kv", 1, "one", 42)
    tdc["config"]["alpha"] = 0.75
    tdc["config"]["experiment_variant"] = "default"
    baseline = _run("h2o", 0, "one", 42)

    rows = aggregate_paper_rows([tdc, baseline])
    tdc_row = next(row for row in rows if row["method"] == "tdc_kv")
    comparisons = paired_significance_rows([tdc, baseline])

    assert tdc_row["variant"] == "default"
    assert len(comparisons) == 1


def test_method_config_identity_prevents_parameter_sweep_collapse():
    first = _run("tdc_kv", 1, "one", 42)
    second = _run("tdc_kv", 1, "two", 42)
    first["method_config_id"] = "a" * 64
    second["method_config_id"] = "b" * 64

    rows = aggregate_paper_rows([first, second])

    assert len(rows) == 2


def test_wilson_interval_is_bounded_and_nonempty():
    low, high = wilson_interval(7, 10)
    assert 0.0 <= low <= 0.7 <= high <= 1.0


def test_paired_significance_does_not_collapse_tdc_parameter_points():
    first = _run("tdc_kv", 1, "one", 42)
    second = _run("tdc_kv", 1, "one", 42)
    baseline = _run("h2o", 0, "one", 42)
    first["method_config_id"] = "a" * 64
    second["method_config_id"] = "b" * 64
    baseline["method_config_id"] = "c" * 64

    comparisons = paired_significance_rows([first, second, baseline])

    assert len(comparisons) == 2
    assert {row["tdc_method_config_id"] for row in comparisons} == {
        "a" * 64,
        "b" * 64,
    }


def test_aggregate_rows_propagate_baseline_fidelity_metadata():
    run = _run("h2o", 1, "one", 42)
    run["_method_metadata"] = {
        "implementation": "local_h2o",
        "reference_equivalence": "approximation",
        "paper_claim_level": "matched_budget_local_approximation",
    }

    row = aggregate_paper_rows([run])[0]

    assert row["reference_equivalence"] == "approximation"
    assert row["paper_claim_level"] == "matched_budget_local_approximation"
