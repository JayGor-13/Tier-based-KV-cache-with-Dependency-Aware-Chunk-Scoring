import torch

from src.core.scorer import DualSignalScorer


def test_forward_single_layer_returns_expected_fused_scores():
    scorer = DualSignalScorer(alpha=0.6, beta=0.4, window_size=2, device="cpu")
    attention_obs = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        ],
        dtype=torch.float32,
    )  # [H=1, w=2, t=4]
    chunks = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
    ]

    score = scorer.forward(attention_obs, chunks)

    assert score.shape == (2,)
    # Both chunks are symmetric under this attention setup.
    assert torch.allclose(score, torch.tensor([0.5, 0.5]), atol=1e-6)


def test_forward_multi_layer_respects_layer_weighting_when_alpha_only():
    scorer = DualSignalScorer(
        alpha=1.0,
        beta=0.0,
        window_size=1,
        num_layers=2,
        device="cpu",
    )
    attention_obs = torch.tensor(
        [
            [[[1.0, 0.0, 0.0]]],  # layer 1 -> M = [1, 0, 0]
            [[[0.0, 1.0, 0.0]]],  # layer 2 -> M = [0, 1, 0]
        ],
        dtype=torch.float32,
    )  # [L=2, H=1, w=1, t=3]
    chunks = [
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
    ]

    score = scorer.forward(attention_obs, chunks)

    # Weighted M = (1/3)*[1,0,0] + (2/3)*[0,1,0] = [1/3, 2/3, 0]
    # Min-max -> [0.5, 1.0, 0.0]
    expected = torch.tensor([0.5, 1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(score, expected, atol=1e-6)


def test_update_scores_only_updates_requested_chunk_indices():
    scorer = DualSignalScorer(alpha=1.0, beta=0.0, window_size=2, device="cpu")
    prev = torch.tensor([0.1, 0.9], dtype=torch.float32)
    attention_obs = torch.tensor(
        [
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    chunks = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
    ]

    updated = scorer.update_scores(
        prev_Score_chunk=prev,
        A_obs_new=attention_obs,
        chunks=chunks,
        updated_chunk_indices=[0],
    )

    assert updated[0] > prev[0]
    # Unaffected chunk index should keep previous value exactly.
    assert torch.isclose(updated[1], prev[1])


def test_signal2_non_window_uses_ema_fallback_across_calls():
    scorer = DualSignalScorer(
        alpha=0.0,
        beta=1.0,
        window_size=2,
        device="cpu",
        r_fallback_mode="ema",
        ema_decay=0.5,
    )

    # Call 1: M = [2,0,0,0,0,0], window tokens (4,5) routed to 2.
    A1 = torch.tensor(
        [[[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]], dtype=torch.float32
    )
    M1, R1 = scorer._compute_signals_single_layer(A1)
    assert torch.allclose(M1, torch.tensor([2, 0, 0, 0, 0, 0], dtype=torch.float32))
    assert torch.allclose(R1, torch.tensor([2, 0, 0, 0, 2, 2], dtype=torch.float32))

    # Call 2: M = [0,2,0,0,0,0].
    # Non-window fallback should become EMA(prev_R, M2) for indices 0..3:
    # 0.5*[2,0,0,0] + 0.5*[0,2,0,0] = [1,1,0,0].
    A2 = torch.tensor(
        [[[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]], dtype=torch.float32
    )
    M2, R2 = scorer._compute_signals_single_layer(A2)
    assert torch.allclose(M2, torch.tensor([0, 2, 0, 0, 0, 0], dtype=torch.float32))
    assert torch.allclose(R2[:4], torch.tensor([1, 1, 0, 0], dtype=torch.float32))
    assert torch.allclose(R2[4:], torch.tensor([2, 2], dtype=torch.float32))


def test_signal2_mass_fallback_matches_m_for_non_window_tokens():
    scorer = DualSignalScorer(
        alpha=0.0,
        beta=1.0,
        window_size=2,
        device="cpu",
        r_fallback_mode="mass",
    )
    A = torch.tensor(
        [[[0.2, 0.3, 0.5, 0.0], [0.1, 0.1, 0.2, 0.6]]], dtype=torch.float32
    )  # t=4, window tokens are 2 and 3.

    M, R = scorer._compute_signals_single_layer(A)
    # Non-window indices [0,1] should equal M under mass fallback.
    assert torch.allclose(R[:2], M[:2])
