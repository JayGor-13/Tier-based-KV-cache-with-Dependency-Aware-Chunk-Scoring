import json
import sys

from benchmarks.eval_metrics import aggregate_grouped_runs, compute_cache_metrics
from benchmarks.hf_runner import resolve_budget_configurations, resolve_budgets
from scripts.aggregate_hf_results import main as aggregate_main


def _successful_run(
    sample_id,
    *,
    sequence_length,
    resolved_budget,
    kept_tokens,
    alpha=0.6,
):
    metrics = compute_cache_metrics(
        sample_id=sample_id,
        original_length=sequence_length,
        budget=resolved_budget,
        kept_length=kept_tokens,
        latency_ms=2.0,
    )
    return {
        "status": "ok",
        "model": "tiny/model",
        "dataset": "gsm8k",
        "method": "tdc_kv",
        "sample_id": sample_id,
        "config": {
            "method": "tdc_kv",
            "budget": resolved_budget,
            "budget_type": "ratio",
            "budget_value": 0.5,
            "theta": 0.3,
            "recent_window": 16,
            "alpha": alpha,
            "beta": 1.0 - alpha,
        },
        "sequence_length": sequence_length,
        "num_chunks": 4,
        "min_chunk_size": 20,
        "max_chunk_size": 40,
        "avg_chunk_size": 25.0,
        "kept_tokens": kept_tokens,
        "tier_counts": {"tier0": 2, "tier1": 1, "tier2": 1},
        "decode_cache_summary": {
            "budget_violations": 0,
            "re_eviction_events": 3,
        },
        "metrics": metrics.to_dict(),
        "evicted_prediction": "42",
        "gold": "#### 42",
    }


def test_ratio_budget_groups_variable_length_samples_together():
    runs = [
        _successful_run(
            "a",
            sequence_length=100,
            resolved_budget=50,
            kept_tokens=48,
        ),
        _successful_run(
            "b",
            sequence_length=150,
            resolved_budget=75,
            kept_tokens=72,
        ),
    ]
    runs.append(
        {
            "status": "error",
            "model": "tiny/model",
            "dataset": "gsm8k",
            "method": "tdc_kv",
            "sample_id": "c",
            "config": dict(runs[0]["config"]),
            "error_type": "RuntimeError",
            "error": "synthetic failure",
        }
    )

    grouped = aggregate_grouped_runs(runs)

    assert len(grouped) == 1
    result = grouped[0]
    assert result["budget"] == {"type": "ratio", "value": 0.5}
    assert result["configuration"]["alpha"] == 0.6
    assert result["run_summary"] == {
        "total": 3,
        "successful": 2,
        "failed": 1,
        "unique_samples": 3,
        "error_types": {"RuntimeError": 1},
    }
    assert result["cache_summary"]["count"] == 2
    assert result["qa_summary"]["final_answer_exact_match"] == 1.0
    assert result["qa_summary"]["primary_metric"] == "gsm8k_accuracy"
    assert result["qa_summary"]["primary_score"] == 1.0
    assert result["sequence_summary"]["sequence_length"]["avg"] == 125.0
    assert result["resolved_budget_summary"]["resolved_budget"]["min"] == 50.0
    assert result["resolved_budget_summary"]["resolved_budget"]["max"] == 75.0
    assert result["decode_cache_summary"]["re_eviction_events"]["sum"] == 6.0
    assert result["chunk_summary"]["max_chunk_size"]["max"] == 40.0


def test_grouped_results_count_empty_predictions_as_incorrect():
    run = _successful_run(
        "empty",
        sequence_length=100,
        resolved_budget=50,
        kept_tokens=50,
    )
    run["evicted_prediction"] = ""

    summary = aggregate_grouped_runs([run])[0]["qa_summary"]

    assert summary["count"] == 1
    assert summary["primary_metric"] == "gsm8k_accuracy"
    assert summary["primary_score"] == 0.0


def test_configuration_values_create_separate_groups():
    runs = [
        _successful_run(
            "a",
            sequence_length=100,
            resolved_budget=50,
            kept_tokens=48,
            alpha=0.6,
        ),
        _successful_run(
            "b",
            sequence_length=100,
            resolved_budget=50,
            kept_tokens=48,
            alpha=0.8,
        ),
    ]

    grouped = aggregate_grouped_runs(runs)

    assert len(grouped) == 2
    assert {result["configuration"]["alpha"] for result in grouped} == {0.6, 0.8}


def test_budget_resolution_preserves_provenance_without_duplicate_work():
    configurations = resolve_budget_configurations(
        100,
        budgets=[50],
        budget_ratios=[0.5],
    )

    assert resolve_budgets(100, budgets=[50], budget_ratios=[0.5]) == [50]
    assert configurations == [
        {
            "budget": 50,
            "descriptor": {
                "type": "combined",
                "value": [
                    {"type": "absolute", "value": 50},
                    {"type": "ratio", "value": 0.5},
                ],
            },
            "specifications": [
                {"type": "absolute", "value": 50},
                {"type": "ratio", "value": 0.5},
            ],
        }
    ]


def test_colliding_budget_specs_share_execution_across_requested_groups():
    run = _successful_run(
        "a",
        sequence_length=100,
        resolved_budget=50,
        kept_tokens=48,
    )
    run["config"]["budget_type"] = "combined"
    run["config"]["budget_value"] = [
        {"type": "absolute", "value": 50},
        {"type": "ratio", "value": 0.5},
    ]
    run["config"]["budget_specifications"] = list(
        run["config"]["budget_value"]
    )

    grouped = aggregate_grouped_runs([run])

    assert len(grouped) == 2
    assert {result["budget"]["type"] for result in grouped} == {
        "absolute",
        "ratio",
    }
    assert all(result["run_summary"]["total"] == 1 for result in grouped)


def test_aggregation_cli_writes_grouped_result_file(tmp_path, monkeypatch):
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "grouped.json"
    input_path.write_text(
        json.dumps(
            {
                "runs": [
                    _successful_run(
                        "a",
                        sequence_length=100,
                        resolved_budget=50,
                        kept_tokens=48,
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_hf_results.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    aggregate_main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["group_count"] == 1
    assert payload["grouped_results"][0]["run_summary"]["successful"] == 1
