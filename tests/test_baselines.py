import torch

from src.baselines.h2o import evict_h2o
from src.baselines.snapkv import evict_snapkv


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
