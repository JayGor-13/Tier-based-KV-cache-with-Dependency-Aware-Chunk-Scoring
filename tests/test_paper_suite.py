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
    assert arguments["dtype"] == "bfloat16"
    assert arguments["attention-mode"] == "all"
    assert arguments["prompt-serialization"] == "raw"
    assert arguments["budget-ratios"] == "0.3,0.2,0.1"


def test_tuning_suite_uses_training_data_not_official_test_records():
    jobs = build_jobs(
        "tuning",
        models="one/model,two/model",
        max_samples=12,
        protocol_manifest="frozen.json",
    )

    _, arguments = jobs[0]
    assert "split=train" in arguments["datasets"]
    assert "partition=tuning" in arguments["datasets"]
    assert "split=test" not in arguments["datasets"]


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


def test_method_override_limits_every_profile_to_tdc_kv():
    jobs = build_jobs(
        "all",
        models="one/model,two/model",
        max_samples=1,
        protocol_manifest="frozen.json",
        methods="tdc_kv",
    )

    assert jobs
    assert {arguments["methods"] for _, arguments in jobs} == {"tdc_kv"}


def test_full_gsm8k_profile_uses_entire_frozen_official_test_split():
    jobs = build_jobs(
        "gsm8k_full",
        models="one/model",
        max_samples=None,
        protocol_manifest="gsm8k_full_manifest.json",
    )

    name, arguments = jobs[0]
    assert name == "gsm8k_full"
    assert arguments["max-samples"] == 1319
    assert "split=test" in arguments["datasets"]
    assert "partition=final" in arguments["datasets"]
    assert arguments["budget-ratios"] == "0.3,0.2,0.1"
    assert arguments["methods"] == "fullkv,streamingllm,h2o,snapkv,chunkkv,tdc_kv"
