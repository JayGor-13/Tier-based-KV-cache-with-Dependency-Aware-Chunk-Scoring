from benchmarks.reproducibility import collect_environment_metadata, seed_everything


def test_environment_metadata_contains_required_provenance():
    seed_everything(17)
    metadata = collect_environment_metadata(seed=17)

    assert metadata["seed"] == 17
    assert metadata["packages"]["torch"]
    assert "available" in metadata["cuda"]
    assert set(metadata["git"]) == {"commit", "branch", "dirty"}
