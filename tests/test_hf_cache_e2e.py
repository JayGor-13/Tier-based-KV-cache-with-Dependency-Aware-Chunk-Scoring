import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from benchmarks import hf_runner
from benchmarks.gsm8k_protocol import CHUNKKV_GSM8K_8SHOT_PROTOCOL
from src.core.evictor import evict_kv_cache
from src.core.numerical import NumericalIntegrityError
from src.models.cache_utils import (
    EvictedGenerationResult,
    HfModelBundle,
    HfGenerationResult,
    apply_repetition_penalty,
    build_position_kwargs,
    extended_rotary_position_capacity,
    generate_text,
    generate_text_with_evicted_cache,
    run_hf_prefill,
)
from src.models.hf_cache_adapter import build_dynamic_cache


class _TokenFixture:
    vocab_size = 48
    pad_token_id = 0
    eos_token_id = -1

    def __call__(self, _prompt, **_kwargs):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(token_id)) for token_id in token_ids)


class _ChatTokenFixture(_TokenFixture):
    chat_template = "fixture-template"

    def __init__(self):
        self.template_calls = 0
        self.tokenizer_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["role"] == "user"
        self.template_calls += 1
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, prompt, **kwargs):
        self.tokenizer_calls.append((prompt, kwargs))
        return super().__call__(prompt, **kwargs)


_MODEL_CACHE = {}


def _model_for_family(family):
    if family in _MODEL_CACHE:
        return _MODEL_CACHE[family]

    torch.manual_seed(19)
    if family == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=48,
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=True,
        )
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config)
    elif family == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_cache=True,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
    elif family == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            use_cache=True,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config)
    else:
        raise AssertionError(f"Unknown test model family: {family}")

    model.eval()
    _MODEL_CACHE[family] = model
    return model


def _prefill(family, *, block_size=3):
    return run_hf_prefill(
        model=_model_for_family(family),
        tokenizer=_TokenFixture(),
        prompt="ignored",
        sample_id=f"{family}_e2e",
        observation_window=2,
        attention_mode="last",
        min_chunk_tokens=1,
        dependency_top_k=None,
        prefill_block_size=block_size,
    )


def _token_granular_eviction(prefill, budget):
    token_count = prefill.sequence_length
    chunks = [torch.tensor([index], dtype=torch.long) for index in range(token_count)]
    scores = torch.linspace(0.0, 1.0, token_count)
    tiers = torch.zeros(token_count, dtype=torch.int8)
    eviction = evict_kv_cache(
        mask_tiers=tiers,
        chunk_scores=scores,
        chunks=chunks,
        k_cache=prefill.k_cache,
        v_cache=prefill.v_cache,
        budget=budget,
        allow_level2_fallback=True,
    )
    return eviction, chunks, scores, tiers


def _dynamic_cache(k_cache, v_cache):
    return build_dynamic_cache(
        [
            (k_cache[layer_index], v_cache[layer_index])
            for layer_index in range(k_cache.shape[0])
        ]
    )


def test_repetition_penalty_matches_transformers_sign_rule():
    logits = torch.tensor([4.0, -4.0, 3.0, -3.0])

    adjusted = apply_repetition_penalty(
        logits,
        torch.tensor([0, 1, 1]),
        penalty=2.0,
    )

    assert torch.equal(adjusted, torch.tensor([2.0, -8.0, 3.0, -3.0]))


def test_nondefault_repetition_penalty_preserves_native_custom_parity():
    model = _model_for_family("qwen2")
    tokenizer = _TokenFixture()
    original_penalty = model.generation_config.repetition_penalty
    model.generation_config.repetition_penalty = 1.2
    try:
        native = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="ignored",
            max_new_tokens=5,
            return_details=True,
        )
        prefill = _prefill("qwen2")
        custom = generate_text_with_evicted_cache(
            model=model,
            tokenizer=tokenizer,
            first_new_token_id=prefill.next_token_id,
            prompt_token_ids=prefill.input_ids,
            max_new_tokens=5,
            k_cache=prefill.k_cache,
            v_cache=prefill.v_cache,
            original_sequence_length=prefill.sequence_length,
            return_details=True,
        )
    finally:
        model.generation_config.repetition_penalty = original_penalty

    assert prefill.next_token_id == native.token_ids[0]
    assert custom.token_ids == native.token_ids
    assert custom.text == native.text


@pytest.mark.parametrize("family", ["gpt2", "llama", "qwen2"])
def test_unpruned_custom_cache_generation_matches_fullkv_token_ids(family):
    model = _model_for_family(family)
    tokenizer = _TokenFixture()
    fullkv = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="ignored",
        max_new_tokens=5,
        max_length=32,
        return_details=True,
    )
    prefill = _prefill(family)
    cache_path = generate_text_with_evicted_cache(
        model=model,
        tokenizer=tokenizer,
        first_new_token_id=prefill.next_token_id,
        max_new_tokens=5,
        k_cache=prefill.k_cache,
        v_cache=prefill.v_cache,
        original_sequence_length=prefill.sequence_length,
        budget=None,
        return_details=True,
    )

    assert isinstance(fullkv, HfGenerationResult)
    assert isinstance(cache_path, EvictedGenerationResult)
    assert cache_path.token_ids == fullkv.token_ids
    assert cache_path.text == fullkv.text


def test_native_generation_rejects_nan_logits_before_token_selection():
    model = _model_for_family("gpt2")
    tokenizer = _TokenFixture()

    def corrupt_logits(_module, _args, output):
        output.logits.fill_(float("nan"))
        return output

    hook = model.register_forward_hook(corrupt_logits)
    try:
        with pytest.raises(NumericalIntegrityError, match="native_generation_logits"):
            generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt="ignored",
                max_new_tokens=2,
            )
    finally:
        hook.remove()


def test_hf_grid_records_protocol_judgment_hashes_and_parity(tmp_path, monkeypatch):
    dataset_path = tmp_path / "gsm8k.jsonl"
    dataset_path.write_text(
        '{"id":"sample-1","question":"What is 2 + 2?","answer":"#### 4"}\n',
        encoding="utf-8",
    )
    model = _model_for_family("qwen2")
    tokenizer = _ChatTokenFixture()
    monkeypatch.setattr(
        hf_runner,
        "load_hf_model_and_tokenizer",
        lambda *_args, **_kwargs: HfModelBundle(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        ),
    )
    spec = hf_runner.DatasetSpec(
        name="gsm8k_chunkkv",
        source=str(dataset_path),
        adapter="gsm8k",
        protocol=CHUNKKV_GSM8K_8SHOT_PROTOCOL,
        prompt_field="question",
        answer_field="answer",
        id_field="id",
    )

    payload = hf_runner.run_hf_grid(
        model_names=["tiny/qwen2"],
        dataset_specs=[spec],
        budgets=[],
        budget_ratios=[],
        thetas=[0.3],
        recent_windows=[2],
        alphas=[0.6],
        methods=["fullkv"],
        max_samples=1,
        max_length=32,
        max_new_tokens=3,
        min_chunk_tokens=1,
        run_fullkv_parity=True,
        prompt_serialization="chat",
    )

    run = payload["runs"][0]
    assert payload["protocols"]["gsm8k_chunkkv"]["shots"] == 8
    assert payload["summary"]["fullkv_parity"]["all_passed"] is True
    assert payload["summary"]["qualification"]["passed"] is True
    assert payload["generation_policies"]["tiny/qwen2"]["do_sample"] is False
    assert run["config"]["generation_policy"]["repetition_penalty"] == 1.0
    assert run["protocol"] == CHUNKKV_GSM8K_8SHOT_PROTOCOL
    assert run["prompt_serialization"] == "chat"
    assert run["config"]["prompt_serialization"] == "chat"
    assert len(run["raw_prompt_sha256"]) == 64
    assert len(run["prompt_sha256"]) == 64
    assert run["raw_prompt_sha256"] != run["prompt_sha256"]
    assert len(run["input_token_sha256"]) == 64
    assert run["generated_token_ids"]
    assert run["judgment"]["judge"] == "gsm8k_final_numeric_exact_match"
    assert run["judgment"]["normalized_gold"] == "4"
    assert run["runtime"]["stages"]["prefill"]["elapsed_ms"] >= 0.0
    assert run["runtime"]["stages"]["decode"]["elapsed_ms"] >= 0.0
    assert run["cache_memory"]["kv_bytes_before"] == run["cache_memory"]["kv_bytes_after"]
    assert payload["environment"]["seed"] == 42
    assert payload["experiment_fingerprint"]
    assert tokenizer.template_calls == 1
    assert len(tokenizer.tokenizer_calls) == 1
    assert tokenizer.tokenizer_calls[0][1]["add_special_tokens"] is False


def test_hf_grid_all_methods_have_fair_scores_runtime_and_resume(tmp_path, monkeypatch):
    dataset_path = tmp_path / "records.jsonl"
    dataset_path.write_text(
        '{"id":"sample-1","prompt":"ignored","answer":"1"}\n',
        encoding="utf-8",
    )
    model = _model_for_family("qwen2")
    tokenizer = _TokenFixture()
    monkeypatch.setattr(
        hf_runner,
        "load_hf_model_and_tokenizer",
        lambda *_args, **_kwargs: HfModelBundle(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        ),
    )
    spec = hf_runner.DatasetSpec(
        name="local",
        source=str(dataset_path),
        prompt_field="prompt",
        answer_field="answer",
        id_field="id",
    )
    checkpoint = tmp_path / "checkpoint.json"
    kwargs = dict(
        model_names=["tiny/qwen2"],
        dataset_specs=[spec],
        budgets=[],
        budget_ratios=[0.5],
        thetas=[0.3],
        recent_windows=[2],
        alphas=[0.6],
        methods=[
            "fullkv",
            "streamingllm",
            "h2o",
            "snapkv",
            "chunkkv",
            "tdc_kv",
        ],
        max_samples=1,
        max_new_tokens=2,
        min_chunk_tokens=1,
        max_chunk_tokens=4,
        prefill_block_size=3,
        checkpoint_path=checkpoint,
    )

    payload = hf_runner.run_hf_grid(**kwargs)
    successful = [row for row in payload["runs"] if row["status"] == "ok"]

    assert checkpoint.exists()
    assert payload["summary"]["qualification"]["passed"] is True
    assert {row["method"] for row in successful} == {
        "fullkv",
        "streamingllm",
        "h2o",
        "snapkv",
        "chunkkv",
        "tdc_kv",
    }
    assert all(row.get("run_key") for row in successful)
    compressed = [row for row in successful if row["method"] != "fullkv"]
    assert all(row["config"]["decode_policy"] == "common_streaming" for row in compressed)
    assert all(row["kept_tokens"] == row["config"]["budget"] for row in compressed)
    assert all(row["runtime"]["stages"]["policy"]["elapsed_ms"] >= 0 for row in compressed)
    chunkkv = next(row for row in compressed if row["method"] == "chunkkv")
    tdc = next(row for row in compressed if row["method"] == "tdc_kv")
    assert chunkkv["score_source"] == "direct_attention_only"
    assert tdc["score_source"] == "attention_plus_dependency_routing"

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["state"] == "complete"
    saved["runs"].append(
        {
            "status": "error",
            "model": "tiny/qwen2",
            "dataset": "local",
            "sample_id": "sample-1",
            "error": "transient",
        }
    )
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")

    resumed = hf_runner.run_hf_grid(**kwargs, resume=True)
    assert all(row["status"] == "ok" for row in resumed["runs"])
    assert [row["run_key"] for row in resumed["runs"] if row.get("run_key")] == [
        row["run_key"] for row in payload["runs"] if row.get("run_key")
    ]

    dataset_path.write_text(
        '{"id":"sample-1","prompt":"changed content","answer":"1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        hf_runner.run_hf_grid(**kwargs, resume=True)


def test_hf_grid_transactional_sqlite_checkpoint_resumes_exact_rows(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "records.jsonl"
    dataset_path.write_text(
        '{"id":"sample-1","prompt":"ignored","answer":"1"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hf_runner,
        "load_hf_model_and_tokenizer",
        lambda *_args, **_kwargs: HfModelBundle(
            model=_model_for_family("qwen2"),
            tokenizer=_TokenFixture(),
            device=torch.device("cpu"),
        ),
    )
    kwargs = dict(
        model_names=["tiny/qwen2"],
        dataset_specs=[
            hf_runner.DatasetSpec(
                name="local",
                source=str(dataset_path),
                prompt_field="prompt",
                answer_field="answer",
                id_field="id",
            )
        ],
        budgets=[],
        budget_ratios=[],
        thetas=[0.3],
        recent_windows=[2],
        alphas=[0.6],
        methods=["fullkv"],
        max_samples=1,
        max_new_tokens=1,
        min_chunk_tokens=1,
        checkpoint_path=tmp_path / "checkpoint.sqlite",
    )

    first = hf_runner.run_hf_grid(**kwargs)
    resumed = hf_runner.run_hf_grid(**kwargs, resume=True)

    assert [row["run_key"] for row in resumed["runs"]] == [
        row["run_key"] for row in first["runs"]
    ]


@pytest.mark.parametrize("family", ["gpt2", "llama", "qwen2"])
def test_compressed_cache_logits_match_independent_forward(family):
    model = _model_for_family(family)
    tokenizer = _TokenFixture()
    prefill = _prefill(family)
    budget = 5
    eviction, chunks, scores, tiers = _token_granular_eviction(prefill, budget)

    assert eviction.kept_indices.numel() == budget
    input_ids = torch.tensor([[prefill.next_token_id]], dtype=torch.long)
    position_ids = torch.tensor([[prefill.sequence_length]], dtype=torch.long)
    reference_kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones(1, budget + 1, dtype=torch.long),
        "past_key_values": _dynamic_cache(
            eviction.new_k_cache,
            eviction.new_v_cache,
        ),
        "use_cache": True,
        "return_dict": True,
    }
    reference_kwargs.update(build_position_kwargs(model, position_ids))
    with extended_rotary_position_capacity(
        model,
        required_sequence_length=prefill.sequence_length + 2,
    ):
        with torch.no_grad():
            reference_logits = model(**reference_kwargs).logits[:, -1, :].detach()

    generated_logits = []

    def capture_logits(_module, _args, output):
        generated_logits.append(output.logits[:, -1, :].detach())

    hook = model.register_forward_hook(capture_logits)
    try:
        result = generate_text_with_evicted_cache(
            model=model,
            tokenizer=tokenizer,
            first_new_token_id=prefill.next_token_id,
            max_new_tokens=2,
            k_cache=eviction.new_k_cache,
            v_cache=eviction.new_v_cache,
            original_sequence_length=prefill.sequence_length,
            budget=budget,
            kept_indices=eviction.kept_indices,
            chunks=chunks,
            chunk_scores=scores,
            mask_tiers=tiers,
            recent_window=2,
            return_details=True,
        )
    finally:
        hook.remove()

    assert isinstance(result, EvictedGenerationResult)
    assert len(generated_logits) == 1
    assert torch.allclose(generated_logits[0], reference_logits, atol=1e-5, rtol=1e-5)
    assert result.cache_summary["budget_violations"] == 0
    assert result.cache_summary["final_cache_tokens"] == budget


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_logical_positions_remain_global_across_prefill_and_decode(family):
    model = _model_for_family(family)
    observed_position_ids = []
    observed_cache_positions = []

    def capture_positions(_module, _args, kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            observed_position_ids.append(
                position_ids.detach().cpu().reshape(-1).tolist()
            )
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            observed_cache_positions.append(
                cache_position.detach().cpu().reshape(-1).tolist()
            )

    hook = model.register_forward_pre_hook(
        capture_positions,
        with_kwargs=True,
    )
    try:
        prefill = _prefill(family, block_size=3)
        eviction, chunks, scores, tiers = _token_granular_eviction(prefill, budget=5)
        generate_text_with_evicted_cache(
            model=model,
            tokenizer=_TokenFixture(),
            first_new_token_id=prefill.next_token_id,
            max_new_tokens=2,
            k_cache=eviction.new_k_cache,
            v_cache=eviction.new_v_cache,
            original_sequence_length=prefill.sequence_length,
            budget=5,
            kept_indices=eviction.kept_indices,
            chunks=chunks,
            chunk_scores=scores,
            mask_tiers=tiers,
            recent_window=2,
        )
    finally:
        hook.remove()

    expected = [[0, 1, 2], [3, 4, 5], [6, 7], [8]]
    assert observed_position_ids == expected
    expected_cache_positions = (
        expected if "cache_position" in build_position_kwargs(
            model,
            torch.tensor([[0]], dtype=torch.long),
        ) else []
    )
    assert observed_cache_positions == expected_cache_positions


def test_repeated_decode_eviction_maintains_exact_budget():
    family = "qwen2"
    prefill = _prefill(family)
    budget = 3
    eviction, chunks, scores, tiers = _token_granular_eviction(prefill, budget)

    assert eviction.new_k_cache.shape[-2] == budget
    assert eviction.new_v_cache.shape[-2] == budget
    result = generate_text_with_evicted_cache(
        model=_model_for_family(family),
        tokenizer=_TokenFixture(),
        first_new_token_id=prefill.next_token_id,
        max_new_tokens=7,
        k_cache=eviction.new_k_cache,
        v_cache=eviction.new_v_cache,
        original_sequence_length=prefill.sequence_length,
        budget=budget,
        kept_indices=eviction.kept_indices,
        chunks=chunks,
        chunk_scores=scores,
        mask_tiers=tiers,
        recent_window=2,
        return_details=True,
    )

    assert isinstance(result, EvictedGenerationResult)
    assert result.cache_summary["generated_tokens"] == 7
    assert result.cache_summary["trim_checks"] == 7
    assert result.cache_summary["re_eviction_events"] == 6
    assert result.cache_summary["total_tokens_removed"] == 6
    assert result.cache_summary["budget_violations"] == 0
    assert result.cache_summary["initial_post_trim_tokens"] == budget
    assert result.cache_summary["max_post_trim_tokens"] == budget
    assert result.cache_summary["final_cache_tokens"] == budget


def test_decode_rejects_nan_logits_before_argmax():
    family = "qwen2"
    model = _model_for_family(family)
    prefill = _prefill(family)

    def corrupt_logits(_module, _args, output):
        output.logits.fill_(float("nan"))
        return output

    hook = model.register_forward_hook(corrupt_logits)
    try:
        with pytest.raises(NumericalIntegrityError, match="decode_logits"):
            generate_text_with_evicted_cache(
                model=model,
                tokenizer=_TokenFixture(),
                first_new_token_id=prefill.next_token_id,
                max_new_tokens=2,
                k_cache=prefill.k_cache,
                v_cache=prefill.v_cache,
                original_sequence_length=prefill.sequence_length,
                budget=None,
                return_details=True,
            )
    finally:
        hook.remove()
