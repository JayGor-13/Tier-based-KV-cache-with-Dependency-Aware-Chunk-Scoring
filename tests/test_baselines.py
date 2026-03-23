import torch

from src.baselines.h2o import evict_h2o


def test_h2o_never_exceeds_budget_when_forced_tokens_are_too_many():
    # seq len 12, but sink+recent protection would force all 12 tokens.
    t = 12
    budget = 6
    attention_obs = torch.rand(2, 4, t, dtype=torch.float32)
    k_cache = torch.randn(3, t, 8)
    v_cache = torch.randn(3, t, 8)

    result = evict_h2o(
        attention_obs=attention_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=budget,
        recent_window=12,
        sink_tokens=4,
        heavy_hitter_ratio=1.0,
    )

    assert int(result.kept_indices.numel()) <= budget
    assert int(result.new_k_cache.shape[-2]) <= budget
    assert int(result.new_v_cache.shape[-2]) <= budget
