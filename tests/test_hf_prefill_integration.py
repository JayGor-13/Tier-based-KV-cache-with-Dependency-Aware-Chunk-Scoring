import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.core.evictor import evict_kv_cache
from src.core.dependency_graph import build_sparse_chunk_dependency_graph
from src.core.masker import assign_protection_tiers
from src.core.scorer import DualSignalScorer
from src.models.cache_utils import (
    EvictedGenerationResult,
    extract_full_kv_cache,
    generate_text_with_evicted_cache,
    run_hf_prefill,
)


class _TinyTokenizer:
    vocab_size = 32
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
        return "." if int(token_ids[0]) in {3, 6} else "x"


def test_real_hf_prefill_reaches_dependency_tiered_kv_eviction():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            use_cache=True,
        )
    ).eval()
    tokenizer = _TinyTokenizer()

    encoded = tokenizer("ignored", return_tensors="pt")
    with torch.no_grad():
        direct = model(
            **encoded,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
    direct_k, direct_v = extract_full_kv_cache(direct.past_key_values)
    direct_next_token = int(torch.argmax(direct.logits[0, -1, :]).item())

    prefill_query_lengths = []

    def record_prefill_queries(_module, _args, kwargs):
        prefill_query_lengths.append(int(kwargs["input_ids"].shape[1]))

    hook = model.register_forward_pre_hook(record_prefill_queries, with_kwargs=True)

    try:
        prefill = run_hf_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt="ignored",
            sample_id="tiny_hf",
            observation_window=2,
            attention_mode="last",
            min_chunk_tokens=1,
            max_chunk_tokens=2,
            dependency_top_k=2,
            prefill_block_size=3,
        )
    finally:
        hook.remove()

    assert prefill_query_lengths == [3, 3, 2]
    assert prefill.prefill_blocks == 3
    assert prefill.prefill_block_size == 3
    assert prefill.max_chunk_tokens == 2
    assert max(chunk.numel() for chunk in prefill.chunks) <= 2
    assert prefill.attention_obs.shape == (2, 2, 8)
    assert prefill.next_token_id == direct_next_token
    assert torch.allclose(
        prefill.attention_obs,
        direct.attentions[-1][0, :, -2:, :].to(torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(prefill.k_cache, direct_k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(prefill.v_cache, direct_v, atol=1e-5, rtol=1e-5)
    direct_graph = build_sparse_chunk_dependency_graph(
        direct.attentions,
        chunk_map=prefill.chunk_map,
        top_k=2,
    )
    assert torch.equal(
        prefill.dependency_graph.neighbor_indices,
        direct_graph.neighbor_indices,
    )
    assert torch.allclose(
        prefill.dependency_graph.edge_weights,
        direct_graph.edge_weights,
        atol=1e-5,
        rtol=1e-5,
    )
    scorer = DualSignalScorer(alpha=0.6, beta=0.4, window_size=2)
    scores = scorer.forward_with_details(
        prefill.attention_obs,
        prefill.chunks,
        dependency_graph=prefill.dependency_graph,
    )
    mask = assign_protection_tiers(
        chunk_scores=scores.chunk_scores,
        protection_scores=scores.dependency_scores,
        chunks=prefill.chunks,
        theta=0.5,
        recent_window=1,
        sequence_length=prefill.sequence_length,
        return_details=True,
    )
    eviction = evict_kv_cache(
        mask_tiers=mask.tiers,
        chunk_scores=scores.chunk_scores,
        chunks=prefill.chunks,
        k_cache=prefill.k_cache,
        v_cache=prefill.v_cache,
        budget=6,
        allow_level2_fallback=True,
    )

    assert prefill.dependency_graph is not None
    assert int((prefill.dependency_graph.neighbor_indices >= 0).sum().item()) > 0
    assert scores.chunk_scores.shape == (len(prefill.chunks),)
    assert mask.tier1_source == "dependency_scores"
    assert eviction.new_k_cache.shape[-2] == eviction.kept_indices.numel()
    assert eviction.new_v_cache.shape == eviction.new_k_cache.shape
    assert eviction.kept_indices.numel() <= 6

    generation = generate_text_with_evicted_cache(
        model=model,
        tokenizer=tokenizer,
        first_new_token_id=prefill.next_token_id,
        max_new_tokens=5,
        k_cache=eviction.new_k_cache,
        v_cache=eviction.new_v_cache,
        original_sequence_length=prefill.sequence_length,
        budget=6,
        kept_indices=eviction.kept_indices,
        chunks=prefill.chunks,
        chunk_scores=scores.chunk_scores,
        mask_tiers=mask.tiers,
        recent_window=1,
        return_details=True,
    )

    assert isinstance(generation, EvictedGenerationResult)
    assert generation.cache_summary["budget_violations"] == 0
    assert generation.cache_summary["max_post_trim_tokens"] <= 6
    assert generation.cache_summary["final_cache_tokens"] <= 6
    assert generation.cache_summary["trim_checks"] == 5
    assert generation.cache_summary["re_eviction_events"] > 0


def test_hf_prefill_rejects_nonpositive_block_size():
    with pytest.raises(ValueError, match="prefill_block_size must be positive"):
        run_hf_prefill(
            model=object(),
            tokenizer=_TinyTokenizer(),
            prompt="ignored",
            sample_id="invalid_block",
            observation_window=2,
            prefill_block_size=0,
        )
