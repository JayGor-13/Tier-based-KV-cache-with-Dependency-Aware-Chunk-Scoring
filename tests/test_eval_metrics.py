from benchmarks.eval_metrics import (
    extract_final_answer,
    final_answer_exact_match,
    summarize_qa,
)


def test_extract_final_answer_prefers_gsm8k_marker():
    gold = "She makes 9 * 2 = $<<9*2=18>>18 per day.\n#### 18"

    assert extract_final_answer(gold) == "18"


def test_final_answer_exact_match_handles_commas_and_currency():
    prediction = "The final answer is $1,300."
    gold = "Some reasoning.\n#### 1300"

    assert final_answer_exact_match(prediction, gold) == 1.0


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

