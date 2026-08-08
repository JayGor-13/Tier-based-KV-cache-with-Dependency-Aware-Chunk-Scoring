import torch

from src.core.masker import assign_protection_tiers
from src.core.evictor import compute_keep_mask


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


def test_dependency_tier1_changes_survival_as_theta_changes():
    chunks = [torch.tensor([i], dtype=torch.long) for i in range(8)]
    fused_scores = torch.tensor(
        [0.9, 0.8, 0.1, 0.7, 0.6, 0.5, 0.4, 0.9], dtype=torch.float32
    )
    dependency_scores = torch.tensor(
        [0.0, 0.1, 1.0, 0.2, 0.3, 0.4, 0.5, 0.0], dtype=torch.float32
    )

    no_soft_tiers = assign_protection_tiers(
        chunk_scores=fused_scores,
        chunks=chunks,
        theta=0.0,
        recent_window=1,
        sequence_length=8,
        protection_scores=dependency_scores,
    )
    dependency_tiers = assign_protection_tiers(
        chunk_scores=fused_scores,
        chunks=chunks,
        theta=0.125,
        recent_window=1,
        sequence_length=8,
        protection_scores=dependency_scores,
    )

    no_soft_keep, _ = compute_keep_mask(
        no_soft_tiers,
        fused_scores,
        chunks,
        sequence_length=8,
        budget=4,
    )
    dependency_keep, _ = compute_keep_mask(
        dependency_tiers,
        fused_scores,
        chunks,
        sequence_length=8,
        budget=4,
    )

    assert not bool(no_soft_keep[2].item())
    assert bool(dependency_keep[2].item())
    assert not torch.equal(no_soft_keep, dependency_keep)


def test_masker_details_report_dependency_tier_source():
    chunks = [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])]
    result = assign_protection_tiers(
        chunk_scores=torch.tensor([0.9, 0.8, 0.1]),
        chunks=chunks,
        theta=0.34,
        recent_window=0,
        sequence_length=3,
        protection_scores=torch.tensor([0.1, 0.2, 1.0]),
        return_details=True,
    )

    assert result.tier1_source == "dependency_scores"
    assert result.tiers[2].item() == 1


def test_theta_protects_exact_fraction_when_dependency_scores_tie():
    chunks = [torch.tensor([i]) for i in range(6)]
    result = assign_protection_tiers(
        chunk_scores=torch.arange(6, dtype=torch.float32),
        chunks=chunks,
        theta=0.5,
        recent_window=1,
        sequence_length=6,
        protection_scores=torch.ones(6),
        return_details=True,
    )

    # Sink chunk 0 and recent chunk 5 are Tier 2. Exactly half of the four
    # remaining candidates receive Tier 1, with stable index tie-breaking.
    assert result.tiers.tolist() == [2, 1, 1, 0, 0, 2]
