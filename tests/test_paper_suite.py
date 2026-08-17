from scripts.run_paper_suite import build_jobs, expand_jobs


def test_main_suite_uses_disjoint_final_partition_and_strict_guards():
    jobs = build_jobs(
        "main",
        models="one/model,two/model",
        max_samples=12,
        protocol_manifest="frozen.json",
    )

    name, arguments = jobs[0]
    assert name == "main"
    assert "partition=final" in arguments["datasets"]
    assert arguments["max-samples"] == 12
    assert arguments["require-no-truncation"] is True
    assert arguments["require-model-preflight"] is True


def test_expensive_suite_jobs_split_by_model_dataset_and_sample_shard():
    jobs = build_jobs(
        "main",
        models="one/model,two/model",
        max_samples=4,
        protocol_manifest="frozen.json",
    )
    expanded = expand_jobs(
        jobs,
        split_models=True,
        split_datasets=True,
        sample_shards=2,
    )

    assert len(expanded) == 2 * 5 * 2
    assert {args["sample-shard-count"] for _, args in expanded} == {2}
    assert {args["sample-shard-index"] for _, args in expanded} == {0, 1}
