from types import SimpleNamespace

import pytest
import torch

from benchmarks.hf_runner import (
    DatasetSpec,
    build_prompt_from_record,
    load_dataset_records,
    parse_dataset_spec,
    run_hf_grid,
    _run_eviction_method,
)
from benchmarks.gsm8k_protocol import CHUNKKV_GSM8K_8SHOT_PROTOCOL


def test_parse_dataset_spec_infers_official_gsm8k_defaults():
    spec = parse_dataset_spec("source=gsm8k,name=math_eval")

    assert spec.name == "math_eval"
    assert spec.source == "openai/gsm8k"
    assert spec.adapter == "gsm8k"
    assert spec.config == "main"
    assert spec.split == "test"
    assert spec.prompt_field == "question"
    assert spec.answer_field == "answer"


def test_gsm8k_adapter_builds_task_prompt_with_final_answer_contract():
    spec = parse_dataset_spec("source=gsm8k")
    prompt, gold = build_prompt_from_record(
        {
            "question": "There are 2 bags with 3 apples each. How many apples?",
            "answer": "Each bag has 3 apples, so 2 * 3 = <<2*3=6>>6.\n#### 6",
        },
        spec,
    )

    assert prompt.startswith("Answer the grade-school math problem.")
    assert "Question: There are 2 bags with 3 apples each." in prompt
    assert "#### <number>" in prompt
    assert gold == "Each bag has 3 apples, so 2 * 3 = <<2*3=6>>6.\n#### 6"


def test_chunkkv_gsm8k_protocol_builds_exact_eight_shot_prompt():
    spec = parse_dataset_spec("source=gsm8k,protocol=gsm8k_chunkkv")
    prompt, gold = build_prompt_from_record(
        {
            "question": "There are 2 bags with 3 apples each. How many apples?",
            "answer": "Reasoning.\n#### 6",
        },
        spec,
    )

    assert spec.protocol == CHUNKKV_GSM8K_8SHOT_PROTOCOL
    assert prompt.count("Question:") == 9
    assert prompt.count("The answer is") == 8
    assert prompt.endswith(
        "Question: There are 2 bags with 3 apples each. How many apples?\n"
    )
    assert gold == "Reasoning.\n#### 6"


def test_paper_protocol_rejects_custom_prompt_templates():
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_dataset_spec(
            "source=gsm8k,protocol=gsm8k_chunkkv,template={prompt}"
        )


def test_hotpotqa_adapter_includes_context_and_exposes_supporting_facts_for_templates():
    spec = parse_dataset_spec("source=hotpot_qa")
    record = {
        "question": "Which city hosted the event?",
        "answer": "Paris",
        "context": {
            "title": ["Event", "Paris"],
            "sentences": [
                ["The event was hosted in the capital of France."],
                ["Paris is the capital of France."],
            ],
        },
        "supporting_facts": {"title": ["Event", "Paris"], "sent_id": [0, 0]},
    }

    prompt, gold = build_prompt_from_record(record, spec)
    templated_prompt, _ = build_prompt_from_record(
        record,
        spec,
        prompt_template="Q: {prompt}\n{context}\nSF: {supporting_facts}",
    )

    assert spec.source == "hotpotqa/hotpot_qa"
    assert spec.adapter == "hotpotqa"
    assert "Context:" in prompt
    assert "[1] Event" in prompt
    assert "The event was hosted in the capital of France." in prompt
    assert "[2] Paris" in prompt
    assert "Question: Which city hosted the event?" in prompt
    assert "SF: Event:0; Paris:0" in templated_prompt
    assert gold == "Paris"


def test_niah_adapter_generates_controlled_context_and_prompt():
    spec = parse_dataset_spec(
        "source=niah,context_length=40,needle_depth=0.25,needle_prefix=KEY,seed=7"
    )

    records = load_dataset_records(spec, max_samples=2)
    prompt, gold = build_prompt_from_record(records[0], spec)
    words = records[0]["context"].split()
    needle_word_index = records[0]["needle_word_index"]

    assert len(records) == 2
    assert len(words) == 40
    assert records[0]["needle"] == "KEY-000007"
    assert words[needle_word_index : needle_word_index + 6] == [
        "The",
        "secret",
        "retrieval",
        "key",
        "is",
        "KEY-000007.",
    ]
    assert gold == "KEY-000007"
    assert "Retrieve the secret key exactly." in prompt
    assert "Question: What is the secret retrieval key?" in prompt


def test_generic_dataset_prompt_path_still_uses_prompt_field_only():
    spec = DatasetSpec(name="custom", source="local.jsonl")

    prompt, gold = build_prompt_from_record(
        {"prompt": "Say hello", "answer": "hello"},
        spec,
    )

    assert prompt == "Say hello"
    assert gold is None


def test_hf_grid_rejects_max_chunk_size_below_minimum():
    with pytest.raises(
        ValueError,
        match="max_chunk_tokens must be greater than or equal",
    ):
        run_hf_grid(
            model_names=["unused/model"],
            dataset_specs=[DatasetSpec(name="unused", source="unused.jsonl")],
            budgets=[],
            budget_ratios=[0.5],
            thetas=[0.3],
            recent_windows=[16],
            alphas=[0.6],
            min_chunk_tokens=8,
            max_chunk_tokens=4,
        )


def test_hf_runner_dependency_tier_mode_changes_bridge_survival():
    chunks = [torch.tensor([i], dtype=torch.long) for i in range(8)]
    prefill = SimpleNamespace(
        chunks=chunks,
        sequence_length=8,
        k_cache=torch.zeros(1, 8, 1),
        v_cache=torch.zeros(1, 8, 1),
    )
    fused_scores = torch.tensor([0.9, 0.8, 0.1, 0.7, 0.6, 0.5, 0.4, 0.9])
    dependency_scores = torch.tensor([0.0, 0.1, 1.0, 0.2, 0.3, 0.4, 0.5, 0.0])
    attention_obs = torch.ones(1, 1, 8)

    dependency_eviction, dependency_mask = _run_eviction_method(
        "tdc_kv",
        prefill,
        budget=4,
        theta=0.125,
        recent_window=1,
        chunk_scores=fused_scores,
        protection_scores=dependency_scores,
        tier1_score_mode="dependency",
        attention_obs=attention_obs,
        allow_level2_fallback=False,
    )
    no_tier1_eviction, no_tier1_mask = _run_eviction_method(
        "tdc_kv",
        prefill,
        budget=4,
        theta=0.125,
        recent_window=1,
        chunk_scores=fused_scores,
        protection_scores=dependency_scores,
        tier1_score_mode="none",
        attention_obs=attention_obs,
        allow_level2_fallback=False,
    )

    assert dependency_mask is not None
    assert no_tier1_mask is not None
    assert dependency_mask.tier1_source == "dependency_scores"
    assert no_tier1_mask.tier1_source == "disabled"
    assert 2 in dependency_eviction.kept_indices.tolist()
    assert 2 not in no_tier1_eviction.kept_indices.tolist()
