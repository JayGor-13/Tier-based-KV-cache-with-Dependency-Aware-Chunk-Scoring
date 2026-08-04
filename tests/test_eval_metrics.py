import pytest

from benchmarks.eval_metrics import (
    compute_cache_metrics,
    extract_final_answer,
    final_answer_exact_match,
    hotpotqa_exact_match,
    hotpotqa_f1,
    judge_gsm8k_prediction,
    niah_retrieval_match,
    normalize_final_answer,
    summarize_generation_parity,
    summarize_qa,
    validate_budget_contract,
)


def test_extract_final_answer_prefers_gsm8k_marker():
    gold = "She makes 9 * 2 = $<<9*2=18>>18 per day.\n#### 18"

    assert extract_final_answer(gold) == "18"


def test_final_answer_exact_match_handles_commas_and_currency():
    prediction = "The final answer is $1,300."
    gold = "Some reasoning.\n#### 1300"

    assert final_answer_exact_match(prediction, gold) == 1.0


@pytest.mark.parametrize(
    ("prediction", "normalized"),
    [
        ("The answer is $1,300.", "1300"),
        ("The answer is -$12.50.", "-12.5"),
        ("The answer is $-12.50.", "-12.5"),
        ("The answer is -0.00.", "0"),
    ],
)
def test_normalize_final_answer_handles_paper_reporting_formats(
    prediction,
    normalized,
):
    assert normalize_final_answer(prediction) == normalized


def test_gsm8k_judgment_is_auditable():
    judgment = judge_gsm8k_prediction(
        "Reasoning. The answer is $1,300.",
        "Gold reasoning.\n#### 1300",
        protocol="chunkkv_gsm8k_8shot",
    )

    assert judgment == {
        "judge": "gsm8k_final_numeric_exact_match",
        "judge_version": 1,
        "protocol": "chunkkv_gsm8k_8shot",
        "normalized_prediction": "1300",
        "normalized_gold": "1300",
        "score": 1.0,
        "correct": True,
    }


def test_generation_parity_summary_requires_text_and_token_matches():
    summary = summarize_generation_parity(
        [
            {"sample_id": "a", "text_match": True, "token_match": True},
            {"sample_id": "b", "text_match": True, "token_match": False},
        ]
    )

    assert summary["text_parity_rate"] == 1.0
    assert summary["token_parity_rate"] == 0.5
    assert summary["all_passed"] is False
    assert summary["mismatched_samples"] == ["b"]


def test_summarize_qa_includes_gsm8k_final_answer_metrics():
    summary = summarize_qa(
        [
            {
                "dataset": "gsm8k",
                "prediction": "The answer is 18.",
                "gold": "Reasoning.\n#### 18",
            },
            {
                "dataset": "gsm8k",
                "prediction": "The answer is 17.",
                "gold": "Reasoning.\n#### 18",
            },
        ]
    )

    assert summary["count"] == 2
    assert summary["final_answer_count"] == 2
    assert summary["final_answer_exact_match"] == 0.5
    assert summary["task_family"] == "gsm8k"
    assert summary["primary_metric"] == "gsm8k_accuracy"
    assert summary["primary_score"] == 0.5


def test_niah_retrieval_accepts_verbose_exact_key_but_not_partial_key():
    assert niah_retrieval_match(
        "The secret retrieval key is NIAH-000013.",
        "NIAH-000013",
    ) == 1.0
    assert niah_retrieval_match("NIAH-0000137", "NIAH-000013") == 0.0


def test_summarize_qa_uses_niah_retrieval_accuracy_as_primary_metric():
    summary = summarize_qa(
        [
            {
                "dataset": "niah_512",
                "prediction": "The key is NIAH-000013.",
                "gold": "NIAH-000013",
            },
            {
                "dataset": "niah_512",
                "prediction": "I could not find it.",
                "gold": "NIAH-000014",
            },
        ]
    )

    assert summary["final_answer_count"] == 0
    assert summary["task_family"] == "niah"
    assert summary["primary_metric"] == "niah_retrieval_accuracy"
    assert summary["primary_score"] == 0.5
    assert summary["dataset_metrics"]["niah"]["retrieval_accuracy"] == 0.5


def test_hotpotqa_metrics_follow_official_answer_normalization():
    assert hotpotqa_exact_match("The Eiffel-Tower", "eiffeltower") == 1.0
    assert hotpotqa_f1("Paris, France", "Paris France") == 1.0
    assert hotpotqa_f1("yes", "no") == 0.0
    assert hotpotqa_f1("yes definitely", "yes") == 0.0
    assert hotpotqa_f1("", "") == 0.0


def test_summarize_qa_uses_hotpotqa_f1_as_primary_metric():
    summary = summarize_qa(
        [
            {
                "dataset": "hotpotqa",
                "prediction": "Paris France",
                "gold": "Paris",
            }
        ]
    )

    assert summary["final_answer_count"] == 0
    assert summary["task_family"] == "hotpotqa"
    assert summary["primary_metric"] == "hotpotqa_f1"
    assert summary["primary_score"] == pytest.approx(2.0 / 3.0)
    assert summary["secondary_metric"] == "hotpotqa_exact_match"
    assert summary["secondary_score"] == 0.0
    assert summary["dataset_metrics"]["hotpotqa"]["precision"] == 0.5
    assert summary["dataset_metrics"]["hotpotqa"]["recall"] == 1.0


def test_mixed_dataset_summary_reports_macro_task_score():
    summary = summarize_qa(
        [
            {"dataset": "gsm8k", "prediction": "42", "gold": "#### 42"},
            {"dataset": "niah", "prediction": "missing", "gold": "KEY-1"},
        ]
    )

    assert summary["task_family"] == "mixed"
    assert summary["primary_metric"] == "macro_task_score"
    assert summary["primary_score"] == 0.5


def test_cache_metrics_report_budget_utilization_and_shortfall():
    metrics = compute_cache_metrics(
        sample_id="sample",
        original_length=100,
        budget=50,
        kept_length=49,
        latency_ms=1.0,
    )

    assert metrics.target_budget == 50
    assert metrics.budget_shortfall == 1
    assert metrics.budget_overflow == 0
    assert metrics.budget_utilization == 0.98


def test_budget_contract_rejects_low_utilization_even_with_small_shortfall():
    metrics = compute_cache_metrics(
        sample_id="short-budget",
        original_length=20,
        budget=10,
        kept_length=9,
        latency_ms=0.0,
    )

    with pytest.raises(ValueError, match="utilization"):
        validate_budget_contract(
            metrics,
            min_utilization=0.99,
            max_shortfall_tokens=1,
        )
