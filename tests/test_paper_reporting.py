from benchmarks.paper_reporting import (
    aggregate_paper_rows,
    paired_significance_rows,
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
