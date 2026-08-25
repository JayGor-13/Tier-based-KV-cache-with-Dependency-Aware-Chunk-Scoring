from benchmarks.eval_metrics import compute_cache_metrics
from scripts.summarize_local_paper_results import aggregate_algorithm_parameter_rows


def _run(sample, *, kept_tokens, prediction="42"):
    metrics = compute_cache_metrics(
        sample_id=sample,
        original_length=100,
        budget=50,
        kept_length=kept_tokens,
        latency_ms=1.0,
    )
    return {
        "status": "ok",
        "run_key": f"run-{sample}",
        "method": "tdc_kv",
        "model": "small/model",
        "dataset": "gsm8k",
        "sample_id": sample,
        "config": {
            "budget_specifications": [{"type": "ratio", "value": 0.5}],
            "theta": 0.3,
            "alpha": 0.6,
            "beta": 0.4,
            "recent_window": 16,
            "chunking_strategy": "sentence",
            "fixed_chunk_size": 16,
            "min_chunk_tokens": 5,
            "max_chunk_tokens": 64,
            "dependency_top_k": 8,
            "tier1_score_mode": "dependency",
            "attention_mode": "last",
            "layer_weighting": "linear",
            "protect_sink": True,
            "protect_recent": True,
            "allow_level2_fallback": True,
            "decode_policy": "common_streaming",
        },
        "sequence_length": 100,
        "kept_tokens": kept_tokens,
        "removed_tokens": 100 - kept_tokens,
        "num_chunks": 4,
        "avg_chunk_size": 25.0,
        "cache_memory": {
            "kv_bytes_before": 1024,
            "kv_bytes_after": 512,
            "kv_bytes_saved": 512,
        },
        "runtime": {
            "method_specific_ms": 3.0,
            "decode_tokens_per_second": 10.0,
            "decode_ms_per_token": 100.0,
            "stages": {
                "prefill": {"elapsed_ms": 1.0},
                "scoring": {"elapsed_ms": 2.0},
                "policy": {"elapsed_ms": 1.0},
                "decode": {"elapsed_ms": 4.0},
            },
        },
        "structural_metrics": {
            "evidence_token_retention": 1.0,
            "evidence_chunk_survival": 1.0,
        },
        "tier_counts": {"tier0": 1, "tier1": 2, "tier2": 3},
        "evicted_prediction": prediction,
        "gold": "#### 42",
        "metrics": metrics.to_dict(),
    }


def test_algorithm_parameter_summary_groups_by_ratio_and_parameters():
    rows = aggregate_algorithm_parameter_rows(
        [_run("a", kept_tokens=50), _run("b", kept_tokens=48)]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "small/model"
    assert row["dataset"] == "gsm8k"
    assert row["method"] == "tdc_kv"
    assert row["requested_retention_ratio"] == 0.5
    assert row["requested_compression_ratio"] == 0.5
    assert row["theta"] == 0.3
    assert row["alpha"] == 0.6
    assert row["recent_window"] == 16
    assert row["metric"] == "gsm8k_accuracy"
    assert row["quality_mean"] == 1.0
    assert row["samples"] == 2
    assert row["actual_retention_ratio"] == 0.49
    assert row["budget_shortfall_max"] == 2


def test_algorithm_parameter_summary_separates_changed_alpha():
    first = _run("a", kept_tokens=50)
    second = _run("b", kept_tokens=50)
    second["config"]["alpha"] = 0.75
    second["config"]["beta"] = 0.25

    rows = aggregate_algorithm_parameter_rows([first, second])

    assert len(rows) == 2
    assert {row["alpha"] for row in rows} == {0.6, 0.75}


def test_algorithm_parameter_summary_separates_protocol_profiles():
    first = _run("a", kept_tokens=50)
    second = _run("b", kept_tokens=50)
    first["config"]["experiment_variant"] = "main"
    second["config"]["experiment_variant"] = "parameter_sweep"

    rows = aggregate_algorithm_parameter_rows([first, second])

    assert len(rows) == 2
    assert {row["experiment_variant"] for row in rows} == {
        "main",
        "parameter_sweep",
    }


def test_algorithm_parameter_summary_rejects_conflicting_duplicate_keys():
    first = _run("a", kept_tokens=50)
    second = _run("a", kept_tokens=48)

    try:
        aggregate_algorithm_parameter_rows([first, second])
    except ValueError as exc:
        assert "Conflicting duplicate" in str(exc)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("conflicting duplicate was not rejected")


def test_algorithm_parameter_summary_reports_nqr_against_fullkv():
    dense = _run("a", kept_tokens=100)
    dense["method"] = "fullkv"
    dense["config"]["budget_specifications"] = []
    compressed = _run("b", kept_tokens=50)

    rows = aggregate_algorithm_parameter_rows([dense, compressed])

    dense_row = next(row for row in rows if row["method"] == "fullkv")
    compressed_row = next(row for row in rows if row["method"] == "tdc_kv")
    assert dense_row["normalized_quality_ratio"] == 100.0
    assert compressed_row["normalized_quality_ratio"] == 100.0
    assert compressed_row["quality_wilson_95ci_low"] is not None


def test_algorithm_parameter_summary_propagates_fidelity_label():
    run = _run("a", kept_tokens=50)
    run["_method_metadata"] = {
        "implementation": "local_tdc",
        "reference_equivalence": "native",
        "paper_claim_level": "proposed_method",
    }

    row = aggregate_algorithm_parameter_rows([run])[0]

    assert row["reference_equivalence"] == "native"
    assert row["paper_claim_level"] == "proposed_method"
