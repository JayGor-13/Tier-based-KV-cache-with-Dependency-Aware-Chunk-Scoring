import torch

from src.core.masker import assign_protection_tiers


def test_assign_protection_tiers_matches_spec_semantics():
    chunks = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([4, 5], dtype=torch.long),
        torch.tensor([6, 7], dtype=torch.long),
    ]
    chunk_scores = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)

    tiers = assign_protection_tiers(
        chunk_scores=chunk_scores,
        chunks=chunks,
        theta=0.5,
        recent_window=2,
        sequence_length=8,
    )

    assert tiers.tolist() == [2, 1, 0, 2]


def test_assign_protection_tiers_theta_zero_disables_tier1():
    chunks = [
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
        torch.tensor([3], dtype=torch.long),
    ]
    chunk_scores = torch.tensor([0.5, 0.7, 0.2, 0.9], dtype=torch.float32)

    tiers = assign_protection_tiers(
        chunk_scores=chunk_scores,
        chunks=chunks,
        theta=0.0,
        recent_window=1,
        sequence_length=4,
    )

    # chunk 0 is sink-protected, chunk 3 is in the recent window.
    assert tiers.tolist() == [2, 0, 0, 2]


def test_assign_protection_tiers_can_return_details():
    chunks = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
    ]
    chunk_scores = torch.tensor([0.1, 0.9], dtype=torch.float32)

    result = assign_protection_tiers(
        chunk_scores=chunk_scores,
        chunks=chunks,
        theta=0.5,
        recent_window=1,
        sequence_length=4,
        return_details=True,
    )

    assert hasattr(result, "tiers")
    assert result.sink_chunk_index == 0
    assert result.recent_start_chunk_index == 1
    assert result.tiers.dtype == torch.int8
