import torch

from src.core.pipeline import TDCKVPipeline


def test_full_pipeline_runs_chunk_to_eviction_end_to_end():
    token_ids = torch.tensor([10, 99, 11, 12, 99, 13], dtype=torch.long)
    # Single-layer observed attention: [H=1, w=2, t=6]
    attention_obs = torch.tensor(
        [
            [
                [0.9, 0.8, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.7, 0.1, 0.1, 0.05, 0.05],
            ]
        ],
        dtype=torch.float32,
    )
    k_cache = torch.randn(2, 6, 4)
    v_cache = torch.randn(2, 6, 4)

    pipeline = TDCKVPipeline(
        punct_ids={99},
        min_chunk_tokens=1,
        alpha=0.6,
        beta=0.4,
        window_size=2,
        theta=0.34,
        recent_window=1,
        device="cpu",
    )

    result = pipeline.run(
        token_ids=token_ids,
        attention_obs=attention_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=4,
    )

    assert len(result.chunks) == 3
    assert result.chunk_map.tolist() == [0, 0, 1, 1, 1, 2]
    assert result.chunk_scores.shape == (3,)
    assert result.mask_tiers.tolist() == [2, 0, 2]
    assert result.eviction.kept_indices.tolist() == [0, 1, 5]
    assert result.eviction.removed_indices.tolist() == [2, 3, 4]
    assert result.eviction.new_k_cache.shape[-2] == 3
    assert result.eviction.new_v_cache.shape[-2] == 3
