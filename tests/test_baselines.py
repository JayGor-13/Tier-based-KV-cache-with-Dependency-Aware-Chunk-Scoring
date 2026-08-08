import torch

from src.baselines.chunkkv import evict_chunkkv
from src.baselines.h2o import evict_h2o, h2o_token_scores
from src.baselines.snapkv import evict_snapkv, snapkv_token_scores
from src.baselines.streamingllm import evict_streamingllm
from benchmarks.pipeline import TraceSample, run_baseline_policy


def _cache(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    k_cache = torch.randn(2, seq_len, 4)
    v_cache = torch.randn(2, seq_len, 4)
    return k_cache, v_cache


def test_h2o_trims_forced_recent_tokens_to_budget():
    seq_len = 6
    budget = 3
    attention_obs = torch.full((1, 3, seq_len), 1.0 / seq_len, dtype=torch.float32)
    k_cache, v_cache = _cache(seq_len)

    result = evict_h2o(
        attention_obs=attention_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=budget,
        recent_window=seq_len,
        sink_tokens=1,
    )

    assert int(result.kept_indices.numel()) == budget
    assert result.new_k_cache.shape[-2] == budget
    assert result.new_v_cache.shape[-2] == budget


def test_snapkv_trims_forced_sink_tokens_to_budget():
    seq_len = 6
    budget = 3
    attention_obs = torch.full((1, 3, seq_len), 1.0 / seq_len, dtype=torch.float32)
    k_cache, v_cache = _cache(seq_len)

    result = evict_snapkv(
        attention_obs=attention_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=budget,
        recent_window=2,
        sink_tokens=seq_len,
    )

    assert int(result.kept_indices.numel()) == budget
    assert result.new_k_cache.shape[-2] == budget
    assert result.new_v_cache.shape[-2] == budget


def test_streamingllm_keeps_sink_and_recent_tokens():
    seq_len = 8
    budget = 4
    k_cache, v_cache = _cache(seq_len)

    result = evict_streamingllm(
        k_cache=k_cache,
        v_cache=v_cache,
        budget=budget,
        num_sink_tokens=2,
    )

    assert result.kept_indices.tolist() == [0, 1, 6, 7]
    assert result.new_k_cache.shape[-2] == budget
    assert result.new_v_cache.shape[-2] == budget


def test_pipeline_runs_streamingllm_baseline():
    seq_len = 8
    budget = 4
    k_cache, v_cache = _cache(seq_len)
    sample = TraceSample(
        sample_id="streaming_sample",
        chunks=[torch.arange(0, 4), torch.arange(4, 8)],
        chunk_scores=torch.tensor([0.1, 0.9]),
        k_cache=k_cache,
        v_cache=v_cache,
    )

    result, metrics = run_baseline_policy(
        sample,
        method="streamingllm",
        budget=budget,
    )

    assert int(result.kept_indices.numel()) == budget
    assert metrics.kept_length == budget


def test_chunkkv_refines_variable_chunks_to_exact_token_budget():
    seq_len = 10
    budget = 6
    k_cache, v_cache = _cache(seq_len)
    chunks = [torch.arange(0, 7), torch.arange(7, 10)]

    result = evict_chunkkv(
        chunk_scores=torch.tensor([0.9, 0.1]),
        chunks=chunks,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=budget,
        sequence_length=seq_len,
    )

    assert result.kept_indices.numel() == budget
    assert result.budget_utilization == 1.0
    assert result.budget_shortfall == 0
    assert result.partially_evicted_chunks == 1


def test_h2o_accepts_precomputed_scores_without_changing_selection():
    k_cache, v_cache = _cache(8)
    attention = torch.rand(2, 3, 8)
    expected = evict_h2o(attention, k_cache, v_cache, budget=4)
    actual = evict_h2o(
        attention,
        k_cache,
        v_cache,
        budget=4,
        token_scores=h2o_token_scores(attention),
    )

    assert torch.equal(actual.kept_indices, expected.kept_indices)


def test_snapkv_accepts_precomputed_scores_without_changing_selection():
    k_cache, v_cache = _cache(8)
    attention = torch.rand(2, 3, 8)
    expected = evict_snapkv(attention, k_cache, v_cache, budget=4, recent_window=3)
    actual = evict_snapkv(
        attention,
        k_cache,
        v_cache,
        budget=4,
        recent_window=3,
        token_scores=snapkv_token_scores(attention, window_size=3),
    )

    assert torch.equal(actual.kept_indices, expected.kept_indices)
