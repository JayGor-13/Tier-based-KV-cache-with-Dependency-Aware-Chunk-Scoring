from scripts.run_local_paper_suite import build_local_jobs, expand_jobs


def test_local_main_defaults_target_two_small_models_and_three_task_families():
    jobs = build_local_jobs("main", max_samples=3)

    name, arguments = jobs[0]
    assert name == "main"
    assert arguments["models"] == (
        "Qwen/Qwen2.5-1.5B-Instruct,"
        "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    )
    assert "name=gsm8k" in arguments["datasets"]
    assert "name=hotpotqa" in arguments["datasets"]
    assert "name=niah_3072_d50" in arguments["datasets"]
    assert arguments["budget-ratios"] == "0.75,0.5,0.25,0.125"
    assert arguments["max-samples"] == 3
    assert arguments["device"] == "cuda"
    assert arguments["dtype"] == "float16"


def test_local_param_sweep_exposes_algorithm_parameters():
    jobs = build_local_jobs("param_sweep", max_samples=2)

    name, arguments = jobs[0]
    assert name == "param_sweep"
    assert arguments["methods"] == "tdc_kv"
    assert arguments["thetas"] == "0.2,0.3,0.4"
    assert arguments["recent-windows"] == "16,32"
    assert arguments["alphas"] == "0.25,0.6,0.75"
    assert arguments["experiment-variant"] == "parameter_sweep"


def test_local_jobs_split_by_model_dataset_and_shard():
    jobs = build_local_jobs(
        "smoke",
        models="one/model,two/model",
        max_samples=1,
    )

    expanded = expand_jobs(
        jobs,
        split_models=True,
        split_datasets=True,
        sample_shards=2,
    )

    assert len(expanded) == 2 * 3 * 2
    assert {args["sample-shard-count"] for _, args in expanded} == {2}
    assert {args["sample-shard-index"] for _, args in expanded} == {0, 1}


def test_frozen_manifest_is_attached_to_every_dataset_spec():
    jobs = build_local_jobs(
        "main",
        models="one/model",
        max_samples=1,
        protocol_manifest="frozen.json",
        use_frozen_manifest=True,
    )

    _, arguments = jobs[0]
    dataset_specs = arguments["datasets"].split(";")
    assert dataset_specs
    assert all("manifest=frozen.json" in spec for spec in dataset_specs)
    assert all("partition=final" in spec for spec in dataset_specs)


def test_all_profile_runs_substantive_sweeps_without_smoke_duplication():
    jobs = build_local_jobs("all", models="one/model", max_samples=1)

    assert [name for name, _ in jobs] == ["main", "param_sweep", "baselines"]
