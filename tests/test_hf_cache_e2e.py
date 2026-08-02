import inspect

import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.core.evictor import evict_kv_cache
from src.models.cache_utils import (
    EvictedGenerationResult,
    extended_rotary_position_capacity,
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
        "position_ids": position_ids,
        "past_key_values": _dynamic_cache(
            eviction.new_k_cache,
            eviction.new_v_cache,
        ),
        "use_cache": True,
        "return_dict": True,
    }
    if "cache_position" in inspect.signature(model.forward).parameters:
        reference_kwargs["cache_position"] = position_ids.reshape(-1)
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
def test_cache_position_tracks_global_positions_across_prefill_and_decode(family):
    model = _model_for_family(family)
    observed_positions = []

    def capture_cache_position(_module, _args, kwargs):
        value = kwargs.get("cache_position")
        if value is not None:
            observed_positions.append(value.detach().cpu().tolist())

    hook = model.register_forward_pre_hook(
        capture_cache_position,
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

    assert observed_positions == [[0, 1, 2], [3, 4, 5], [6, 7], [8]]


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
