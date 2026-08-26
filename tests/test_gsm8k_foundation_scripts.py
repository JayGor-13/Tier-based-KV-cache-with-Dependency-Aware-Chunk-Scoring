import json
from types import SimpleNamespace

import pytest
import torch

import scripts.run_gsm8k_compression as compression_script
import scripts.test_gsm8k_answers as answer_script
from scripts.run_gsm8k_compression import parse_args as parse_compression_args
from scripts.run_gsm8k_compression import summarize_runs
from scripts.test_gsm8k_answers import parse_args as parse_answer_args
from scripts.test_gsm8k_answers import summarize_records
from src.models.cache_utils import HfGenerationResult


def test_native_answer_smoke_defaults_to_small_direct_test():
    args = parse_answer_args([])

    assert args.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.samples == 5
    assert args.prompt == "direct"
    assert args.require_nonzero_accuracy is True


def test_native_answer_summary_tracks_parse_rate_accuracy_and_speed():
    summary = summarize_records(
        [
            {
                "prediction": "#### 4",
                "parseable": True,
                "correct": True,
                "generated_tokens": 8,
                "latency_ms": 100.0,
                "prompt_truncated": False,
            },
            {
                "prediction": "",
                "parseable": False,
                "correct": False,
                "generated_tokens": 0,
                "latency_ms": 100.0,
                "prompt_truncated": True,
            },
        ]
    )

    assert summary["parse_rate"] == 0.5
    assert summary["accuracy"] == 0.5
    assert summary["generation_tokens_per_second"] == 40.0
    assert summary["truncated_prompts"] == 1


def test_compression_cli_uses_explicit_retention_ratios():
    args = parse_compression_args(["--retention-ratios", "0.5,0.3"])

    assert args.retention_ratios == [0.5, 0.3]
    assert args.native_parity is True

    with pytest.raises(SystemExit):
        parse_compression_args(["--retention-ratios", "1.1"])


def test_compression_summary_separates_fullkv_and_each_retention_ratio():
    common_runtime = {
        "total_measured_ms": 100.0,
        "decode_tokens_per_second": 20.0,
    }
    rows = [
        {
            "status": "ok",
            "method": "fullkv",
            "config": {"budget_type": "fullkv", "budget_value": None},
            "judgment": {"normalized_prediction": "4", "score": 1.0},
            "runtime": common_runtime,
            "metrics": {"retention_ratio": 1.0},
        },
        {
            "status": "ok",
            "method": "tdc_kv",
            "config": {"budget_type": "ratio", "budget_value": 0.5},
            "judgment": {"normalized_prediction": "5", "score": 0.0},
            "runtime": common_runtime,
            "metrics": {"retention_ratio": 0.5},
        },
        {
            "status": "error",
            "method": "tdc_kv",
            "config": {"budget_type": "ratio", "budget_value": 0.3},
        },
    ]

    summary = summarize_runs(rows)

    assert [(row["method"], row["retention_ratio"]) for row in summary] == [
        ("fullkv", None),
        ("tdc_kv", 0.5),
    ]
    assert summary[0]["accuracy"] == 1.0
    assert summary[1]["accuracy"] == 0.0
    assert summary[1]["mean_actual_retention"] == 0.5


def test_native_answer_main_saves_raw_prompt_and_passes_nonzero_gate(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "native.json"
    fake_model = SimpleNamespace(
        parameters=lambda: iter([torch.nn.Parameter(torch.zeros(2))]),
        config=SimpleNamespace(_commit_hash="model-commit"),
    )
    fake_bundle = SimpleNamespace(
        model=fake_model,
        tokenizer=object(),
        device=torch.device("cpu"),
    )
    fake_prepared = SimpleNamespace(
        rendered_text="rendered prompt",
        input_ids=torch.tensor([[1, 2, 3]]),
        original_token_count=3,
        was_truncated=False,
        serialization="raw",
    )
    monkeypatch.setattr(
        answer_script,
        "load_dataset_records",
        lambda spec, max_samples: [
            {"question": "What is 2 + 2?", "answer": "#### 4"}
        ],
    )
    monkeypatch.setattr(
        answer_script,
        "load_hf_model_and_tokenizer",
        lambda *args, **kwargs: fake_bundle,
    )
    monkeypatch.setattr(
        answer_script,
        "prepare_prompt",
        lambda **kwargs: fake_prepared,
    )
    monkeypatch.setattr(
        answer_script,
        "generate_text",
        lambda **kwargs: HfGenerationResult(text="Reasoning. #### 4", token_ids=(4,)),
    )

    exit_code = answer_script.main(
        ["--model", "fake/model", "--samples", "1", "--output", str(output)]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["summary"]["accuracy"] == 1.0
    assert payload["generation"]["policy"]["repetition_penalty"] == 1.0
    assert payload["records"][0]["raw_prompt"].endswith("Answer:")
    assert payload["records"][0]["rendered_prompt"] == "rendered prompt"


def test_compression_main_saves_three_expected_groups(monkeypatch, tmp_path):
    output = tmp_path / "compression.json"
    runtime = {
        "total_measured_ms": 10.0,
        "decode_tokens_per_second": 5.0,
    }
    runs = [
        {
            "status": "ok",
            "method": "fullkv",
            "config": {"budget_type": "fullkv", "budget_value": None},
            "judgment": {"normalized_prediction": "4", "score": 1.0},
            "runtime": runtime,
        },
        *[
            {
                "status": "ok",
                "method": "tdc_kv",
                "config": {"budget_type": "ratio", "budget_value": ratio},
                "judgment": {"normalized_prediction": "4", "score": 1.0},
                "runtime": runtime,
            }
            for ratio in (0.5, 0.3)
        ],
    ]
    monkeypatch.setattr(
        compression_script,
        "run_hf_grid",
        lambda **kwargs: {
            "runs": runs,
            "summary": {"fullkv_parity": {"all_passed": True}},
        },
    )

    exit_code = compression_script.main(
        ["--model", "fake/model", "--samples", "1", "--output", str(output)]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(payload["gsm8k_foundation_summary"]) == 3
