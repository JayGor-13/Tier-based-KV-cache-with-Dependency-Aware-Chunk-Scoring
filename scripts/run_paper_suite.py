"""Launch the complete resumable TDC-KV paper experiment suite.

Examples:
    python scripts/run_paper_suite.py --profile main
    python scripts/run_paper_suite.py --profile all --resume
    python scripts/run_paper_suite.py --profile main --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tuning import select_from_file
from benchmarks.experiment_identity import identity_sha256
from benchmarks.io_utils import write_json_atomic
from benchmarks.qualification import recompute_declared_qualification
from benchmarks.result_schema import assert_result_payload
DEFAULT_MODELS = (
    "meta-llama/Meta-Llama-3-8B-Instruct,"
    "mistralai/Mistral-7B-Instruct-v0.3,"
    "Qwen/Qwen2-7B-Instruct"
)
METHODS = "fullkv,streamingllm,h2o,snapkv,chunkkv,tdc_kv"
MAIN_DATASETS = ";".join(
    (
        "name=gsm8k,source=openai/gsm8k,config=main,split=test,"
        "adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,"
        "prompt_field=question,answer_field=answer",
        "name=hotpotqa,source=hotpotqa/hotpot_qa,config=distractor,"
        "split=validation,adapter=hotpotqa,prompt_field=question,"
        "answer_field=answer,id_field=id",
        "name=niah_8k_d10,source=niah,adapter=niah,context_length=8192,"
        "needle_depth=0.1,seed=13",
        "name=niah_8k_d50,source=niah,adapter=niah,context_length=8192,"
        "needle_depth=0.5,seed=29",
        "name=niah_8k_d90,source=niah,adapter=niah,context_length=8192,"
        "needle_depth=0.9,seed=47",
    )
)
QUALIFICATION_DATASETS = ";".join(
    (
        "name=gsm8k_qualification,source=openai/gsm8k,config=main,split=train,"
        "adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,"
        "prompt_field=question,answer_field=answer",
        "name=hotpotqa_qualification,source=hotpotqa/hotpot_qa,config=distractor,"
        "split=train,adapter=hotpotqa,prompt_field=question,"
        "answer_field=answer,id_field=id",
        "name=niah_qualification_1k_d50,source=niah,adapter=niah,context_length=1024,"
        "needle_depth=0.5,seed=29",
    )
)
TUNING_DATASETS = ";".join(
    (
        "name=gsm8k_tuning,source=openai/gsm8k,config=main,split=train,"
        "adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,"
        "prompt_field=question,answer_field=answer",
        "name=hotpotqa_tuning,source=hotpotqa/hotpot_qa,config=distractor,"
        "split=train,adapter=hotpotqa,prompt_field=question,"
        "answer_field=answer,id_field=id",
        "name=niah_tuning_8k_d50,source=niah,adapter=niah,context_length=8192,"
        "needle_depth=0.5,seed=71",
    )
)
ABLATION_DATASETS = ";".join(
    (
        "name=hotpotqa,source=hotpotqa/hotpot_qa,config=distractor,"
        "split=validation,adapter=hotpotqa,prompt_field=question,"
        "answer_field=answer,id_field=id",
        "name=niah_8k_d50,source=niah,adapter=niah,context_length=8192,"
        "needle_depth=0.5,seed=29",
    )
)
GSM8K_FULL_DATASET = (
    "name=gsm8k_full,source=openai/gsm8k,config=main,split=test,"
    "adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,"
    "prompt_field=question,answer_field=answer"
)


def _attach_frozen_partition(
    datasets: str,
    *,
    manifest: str,
    partition: str,
) -> str:
    return ";".join(
        f"{spec},manifest={manifest},partition={partition}"
        for spec in datasets.split(";")
        if spec.strip()
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _dataset_name(specification: str) -> str:
    values = {
        key.strip(): value.strip()
        for part in specification.split(",")
        if "=" in part
        for key, value in [part.split("=", 1)]
    }
    return values.get("name", values.get("source", "dataset"))


def _parse_model_revisions(value: str) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for item in value.split(";"):
        if not item.strip():
            continue
        model, separator, revision = item.partition("=")
        if not separator or not model.strip() or not revision.strip():
            raise ValueError("Model revision pins must use model=revision entries.")
        revisions[model.strip()] = revision.strip()
    return revisions


def _base_args(*, models: str, datasets: str, samples: int) -> dict[str, Any]:
    return {
        "models": models,
        "datasets": datasets,
        "methods": METHODS,
        "budget-ratios": "0.3,0.2,0.1",
        "thetas": "0.3",
        "recent-windows": "16",
        "alphas": "0.6",
        "max-chunk-tokens": 64,
        "min-chunk-tokens": 5,
        "dependency-top-k": 8,
        "prefill-block-size": 32,
        "max-samples": samples,
        "max-length": 9216,
        "max-new-tokens": 512,
        "device": "cuda",
        "dtype": "bfloat16",
        "attention-mode": "all",
        "layer-weighting": "linear",
        "decode-policy": "common_streaming",
        "prompt-serialization": "raw",
        "truncation-side": "right",
        "parity-max-samples": 10,
        "require-qualified": True,
        "require-cuda": True,
        "require-frozen-manifest": True,
        "require-model-revision": True,
        "require-driver": True,
        "require-clean-git": True,
        "require-model-preflight": True,
        "require-exact-niah": True,
        "require-no-truncation": True,
    }


def build_jobs(
    profile: str,
    *,
    models: str,
    max_samples: int | None,
    protocol_manifest: str = "protocol/paper_dataset_manifest.json",
    methods: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    if profile in {"qualification", "all"}:
        args = _base_args(
            models=models,
            datasets=_attach_frozen_partition(
                QUALIFICATION_DATASETS,
                manifest=protocol_manifest,
                partition="qualification",
            ),
            samples=max_samples or 10,
        )
        args.update(
            {
                "budget-ratios": "0.5,0.25",
                "max-length": 2048,
                "max-new-tokens": 512,
                "methods": METHODS,
                "prefill-block-size": 16,
                "seed": 42,
                "experiment-variant": "qualification",
            }
        )
        jobs.append(("qualification", args))

    if profile in {"tuning", "all"}:
        args = _base_args(
            models=models.split(",")[0],
            datasets=_attach_frozen_partition(
                TUNING_DATASETS,
                manifest=protocol_manifest,
                partition="tuning",
            ),
            samples=max_samples or 20,
        )
        args.update(
            {
                "methods": "fullkv,tdc_kv",
                "budget-ratios": "0.3,0.1",
                "thetas": "0.2,0.3,0.4",
                "recent-windows": "16,32",
                "alphas": "0.25,0.5,0.6,0.75,1.0",
                "seed": 42,
                "experiment-variant": "tuning",
            }
        )
        jobs.append(("tuning", args))

    if profile in {"main", "all"}:
        args = _base_args(
            models=models,
            datasets=_attach_frozen_partition(
                MAIN_DATASETS,
                manifest=protocol_manifest,
                partition="final",
            ),
            samples=max_samples or 200,
        )
        args.update({"seed": 42, "experiment-variant": "default"})
        jobs.append(("main", args))

    if profile == "gsm8k_full":
        args = _base_args(
            models=models,
            datasets=_attach_frozen_partition(
                GSM8K_FULL_DATASET,
                manifest=protocol_manifest,
                partition="final",
            ),
            samples=max_samples or 1319,
        )
        args.update(
            {
                "max-length": 2048,
                "seed": 42,
                "experiment-variant": "gsm8k_full",
            }
        )
        jobs.append(("gsm8k_full", args))

    if profile in {"timing", "all"}:
        for repetition, seed in enumerate((13, 42, 101), start=1):
            args = _base_args(
                models=models,
                datasets=_attach_frozen_partition(
                    "name=niah_8k_d50,source=niah,adapter=niah,context_length=8192,"
                    "needle_depth=0.5,seed=29",
                    manifest=protocol_manifest,
                    partition="tuning",
                ),
                samples=max_samples or 20,
            )
            args.update(
                {
                    "budget-ratios": "0.25",
                    "seed": seed,
                    "experiment-variant": "default",
                }
            )
            jobs.append((f"timing_rep_{repetition}", args))

    if profile in {"ablations", "all"}:
        base = _base_args(
            models=models.split(",")[0],
            datasets=_attach_frozen_partition(
                ABLATION_DATASETS,
                manifest=protocol_manifest,
                partition="tuning",
            ),
            samples=max_samples or 50,
        )
        base.update(
            {
                "methods": "fullkv,tdc_kv",
                "budget-ratios": "0.25,0.125",
                "max-new-tokens": 64,
                "seed": 42,
            }
        )
        variants = {
            "signal": {"alphas": "0.0,0.25,0.5,0.6,0.75,1.0"},
            "no_tier1": {"tier1-score-mode": "none"},
            "no_sink": {"protect-sink": False},
            "no_recent": {"protect-recent": False},
            "fixed_8": {"chunking-strategy": "fixed", "fixed-chunk-size": 8},
            "fixed_16": {"chunking-strategy": "fixed", "fixed-chunk-size": 16},
            "fixed_32": {"chunking-strategy": "fixed", "fixed-chunk-size": 32},
            "token_level": {"chunking-strategy": "token"},
            "layers_uniform": {
                "attention-mode": "all",
                "layer-weighting": "uniform",
            },
            "layers_linear": {
                "attention-mode": "all",
                "layer-weighting": "linear",
            },
        }
        for name, overrides in variants.items():
            args = dict(base)
            args.update(overrides)
            args["experiment-variant"] = name
            jobs.append((f"ablation_{name}", args))

    if methods is not None:
        requested_methods = ",".join(
            method.strip() for method in methods.split(",") if method.strip()
        )
        if not requested_methods:
            raise ValueError("At least one experiment method must be selected.")
        for _, arguments in jobs:
            arguments["methods"] = requested_methods

    return jobs


def expand_jobs(
    jobs: list[tuple[str, dict[str, Any]]],
    *,
    split_models: bool,
    split_datasets: bool,
    sample_shards: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Split expensive execution profiles into independently resumable jobs."""
    expanded: list[tuple[str, dict[str, Any]]] = []
    for name, arguments in jobs:
        expensive = name in {"main", "gsm8k_full"} or name.startswith(
            ("timing_", "ablation_")
        )
        if not expensive:
            expanded.append((name, arguments))
            continue
        model_groups = (
            [model.strip() for model in str(arguments["models"]).split(",") if model.strip()]
            if split_models
            else [str(arguments["models"])]
        )
        dataset_groups = (
            [dataset.strip() for dataset in str(arguments["datasets"]).split(";") if dataset.strip()]
            if split_datasets
            else [str(arguments["datasets"])]
        )
        for model in model_groups:
            for dataset in dataset_groups:
                for shard in range(max(1, int(sample_shards))):
                    split_arguments = dict(arguments)
                    split_arguments["models"] = model
                    split_arguments["datasets"] = dataset
                    split_arguments["sample-shard-count"] = max(1, int(sample_shards))
                    split_arguments["sample-shard-index"] = shard
                    suffix = []
                    if split_models:
                        suffix.append(_slug(model))
                    if split_datasets:
                        suffix.append(_slug(_dataset_name(dataset)))
                    if sample_shards > 1:
                        suffix.append(f"shard_{shard + 1}_of_{sample_shards}")
                    expanded_name = "_".join((name, *suffix)) if suffix else name
                    expanded.append((expanded_name, split_arguments))
    return expanded


def _command(
    python: str,
    args: dict[str, Any],
    *,
    output: Path,
    checkpoint: Path,
    resume: bool,
) -> list[str]:
    command = [python, str(ROOT / "scripts" / "run_hf_grid.py")]
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            command.append(flag if value else f"--no-{key}")
        elif value is not None:
            command.extend((flag, str(value)))
    command.extend(("--checkpoint", str(checkpoint), "--output", str(output)))
    if resume and checkpoint.exists():
        command.append("--resume")
    return command


def _apply_selected_configuration(
    jobs: list[tuple[str, dict[str, Any]]],
    configuration: dict[str, Any],
) -> None:
    for name, arguments in jobs:
        if name.startswith(("qualification", "tuning")):
            continue
        arguments["thetas"] = str(configuration["theta"])
        arguments["recent-windows"] = str(configuration["recent_window"])
        if not name.startswith("ablation_signal"):
            arguments["alphas"] = str(configuration["alpha"])
        if not any(token in name for token in ("fixed_", "token_level")):
            arguments["chunking-strategy"] = configuration["chunking_strategy"]
            arguments["fixed-chunk-size"] = configuration["fixed_chunk_size"]
        if "layers_" not in name:
            arguments["layer-weighting"] = configuration["layer_weighting"]
        if not name.startswith("ablation_no_tier1"):
            arguments["tier1-score-mode"] = configuration["tier1_score_mode"]
        if not name.startswith("ablation_no_sink"):
            arguments["protect-sink"] = configuration["protect_sink"]
        if not name.startswith("ablation_no_recent"):
            arguments["protect-recent"] = configuration["protect_recent"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "qualification",
            "tuning",
            "main",
            "gsm8k_full",
            "timing",
            "ablations",
            "all",
        ),
        default="main",
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--methods",
        default=None,
        help=(
            "Optional comma-separated method override for every selected profile. "
            "Use --methods tdc_kv when baseline results already exist."
        ),
    )
    parser.add_argument("--model-revisions", default="")
    parser.add_argument(
        "--model-revisions-file",
        default=None,
        help="JSON produced by scripts/preflight_hf_models.py",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/paper")
    parser.add_argument(
        "--protocol-manifest",
        default="protocol/paper_dataset_manifest.json",
    )
    parser.add_argument("--freeze-manifest", action="store_true")
    parser.add_argument("--sample-shards", type=int, default=1)
    parser.add_argument(
        "--split-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--split-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--selected-config",
        default=None,
        help="Frozen selected_config.json to apply to main/ablation jobs",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    output_root = Path(options.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_manifest = Path(options.protocol_manifest).resolve()
    if options.freeze_manifest:
        unique_specs: dict[str, str] = {}
        freeze_datasets = (
            GSM8K_FULL_DATASET
            if options.profile == "gsm8k_full"
            else f"{QUALIFICATION_DATASETS};{TUNING_DATASETS};{MAIN_DATASETS}"
        )
        for specification in freeze_datasets.split(";"):
            name = _dataset_name(specification)
            previous = unique_specs.get(name)
            if previous is not None and previous != specification:
                raise ValueError(f"Conflicting frozen dataset specs for `{name}`.")
            unique_specs[name] = specification
        freeze_command = [
            options.python,
            str(ROOT / "scripts" / "freeze_dataset_manifests.py"),
            "--datasets",
            ";".join(unique_specs.values()),
            "--output",
            str(protocol_manifest),
        ]
        if options.profile == "gsm8k_full":
            freeze_command.extend(("--partitions", "final=1319"))
        completed = subprocess.run(freeze_command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        print(
            "Dataset protocol frozen. Review and commit the manifest before "
            "running a clean-worktree paper job.",
            flush=True,
        )
        return
    if not options.dry_run and not protocol_manifest.exists():
        raise FileNotFoundError(
            f"Frozen protocol manifest is missing: {protocol_manifest}. "
            "Run with --freeze-manifest once before execution."
        )
    jobs = build_jobs(
        options.profile,
        models=options.models,
        max_samples=options.max_samples,
        protocol_manifest=str(protocol_manifest),
        methods=options.methods,
    )
    jobs = expand_jobs(
        jobs,
        split_models=options.split_models,
        split_datasets=options.split_datasets,
        sample_shards=options.sample_shards,
    )
    if options.model_revisions and options.model_revisions_file:
        raise ValueError("Use either --model-revisions or --model-revisions-file.")
    model_revisions_value = options.model_revisions
    if options.model_revisions_file:
        revision_payload = json.loads(
            Path(options.model_revisions_file).read_text(encoding="utf-8")
        )
        revision_mapping = revision_payload.get("models") or {}
        model_revisions_value = ";".join(
            f"{model}={revision}" for model, revision in revision_mapping.items()
        )
    if model_revisions_value:
        for _, arguments in jobs:
            arguments["model-revisions"] = model_revisions_value
    if not options.dry_run:
        revision_mapping = _parse_model_revisions(model_revisions_value)
        required_models = {
            model.strip()
            for _, arguments in jobs
            for model in str(arguments["models"]).split(",")
            if model.strip()
        }
        missing_revisions = sorted(required_models - revision_mapping.keys())
        if missing_revisions:
            raise ValueError(
                "Immutable model revision pins are required for every job: "
                f"{missing_revisions}. Run scripts/preflight_hf_models.py first."
            )
    selected_configuration = None
    if options.selected_config:
        selected_payload = json.loads(
            Path(options.selected_config).read_text(encoding="utf-8")
        )
        selected_configuration = selected_payload["selected"]["configuration"]
        _apply_selected_configuration(jobs, selected_configuration)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": options.profile,
        "selected_configuration": selected_configuration,
        "jobs": [],
    }
    for job_name, arguments in jobs:
        job_request_id = identity_sha256(arguments)
        arguments["orchestration-job-id"] = job_request_id
        output = output_root / f"{job_name}.json"
        checkpoint = output_root / f"{job_name}.checkpoint.sqlite"
        legacy_checkpoint = output_root / f"{job_name}.checkpoint.json"
        if options.resume and not checkpoint.exists() and legacy_checkpoint.exists():
            checkpoint = legacy_checkpoint
        if options.resume and output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            qualified = False
            try:
                assert_result_payload(existing, require_complete=True)
                recomputed = recompute_declared_qualification(existing)
                qualified = (
                    recomputed.get("passed") is True
                    and existing.get("orchestration_job_id") == job_request_id
                )
            except (TypeError, ValueError):
                qualified = False
            if qualified:
                job_record = {
                    "name": job_name,
                    "output": str(output),
                    "checkpoint": str(checkpoint),
                    "status": "skipped_complete",
                }
                manifest["jobs"].append(job_record)
                if job_name == "tuning" and options.profile == "all":
                    selected_path = output_root / "selected_config.json"
                    selection = select_from_file(output, selected_path)
                    selected_configuration = selection["selected"]["configuration"]
                    _apply_selected_configuration(jobs, selected_configuration)
                    manifest["selected_configuration"] = selected_configuration
                    manifest["selected_config_path"] = str(selected_path)
                write_json_atomic(output_root / "suite_manifest.json", manifest)
                print(f"\n[{job_name}] already complete; skipping.", flush=True)
                continue
        command = _command(
            options.python,
            arguments,
            output=output,
            checkpoint=checkpoint,
            resume=options.resume,
        )
        printable = subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)
        print(f"\n[{job_name}] {printable}", flush=True)
        job_record = {
            "name": job_name,
            "output": str(output),
            "checkpoint": str(checkpoint),
            "command": command,
            "status": "planned" if options.dry_run else "running",
        }
        manifest["jobs"].append(job_record)
        write_json_atomic(output_root / "suite_manifest.json", manifest)
        if options.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        job_record["return_code"] = int(completed.returncode)
        job_record["status"] = "complete" if completed.returncode == 0 else "failed"
        write_json_atomic(output_root / "suite_manifest.json", manifest)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        if job_name == "tuning" and options.profile == "all":
            selected_path = output_root / "selected_config.json"
            selection = select_from_file(output, selected_path)
            selected_configuration = selection["selected"]["configuration"]
            _apply_selected_configuration(jobs, selected_configuration)
            manifest["selected_configuration"] = selected_configuration
            manifest["selected_config_path"] = str(selected_path)
            write_json_atomic(output_root / "suite_manifest.json", manifest)

    print(f"\nSuite manifest: {output_root / 'suite_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
