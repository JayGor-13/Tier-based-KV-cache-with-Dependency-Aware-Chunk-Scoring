"""Run a laptop-friendly TDC-KV paper sweep on 1-2B models.

This wrapper keeps the existing HuggingFace runner as the single execution
engine and only standardizes model, dataset, parameter, checkpoint, and summary
layout for local RTX 4000-series experiments.

Examples:
    python scripts/run_local_paper_suite.py --profile smoke --dry-run
    python scripts/run_local_paper_suite.py --profile main --resume
    python scripts/run_local_paper_suite.py --profile param_sweep --max-samples 20
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


DEFAULT_LOCAL_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct,"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
)
DEFAULT_METHODS = "fullkv,tdc_kv"
BASELINE_METHODS = "fullkv,streamingllm,h2o,snapkv,chunkkv,tdc_kv"
DEFAULT_BUDGET_RATIOS = "0.75,0.5,0.25,0.125"
DEFAULT_PARAM_BUDGET_RATIOS = "0.5,0.25,0.125"
DEFAULT_THETAS = "0.3"
DEFAULT_PARAM_THETAS = "0.2,0.3,0.4"
DEFAULT_RECENT_WINDOWS = "16"
DEFAULT_PARAM_RECENT_WINDOWS = "16,32"
DEFAULT_ALPHAS = "0.6"
DEFAULT_PARAM_ALPHAS = "0.25,0.6,0.75"


def _main_datasets(niah_context_length: int) -> str:
    return ";".join(
        (
            "name=gsm8k,source=openai/gsm8k,config=main,split=test,"
            "adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,"
            "prompt_field=question,answer_field=answer",
            "name=hotpotqa,source=hotpotqa/hotpot_qa,config=distractor,"
            "split=validation,adapter=hotpotqa,prompt_field=question,"
            "answer_field=answer,id_field=id",
            f"name=niah_{niah_context_length}_d10,source=niah,adapter=niah,"
            f"context_length={niah_context_length},needle_depth=0.1,seed=13",
            f"name=niah_{niah_context_length}_d50,source=niah,adapter=niah,"
            f"context_length={niah_context_length},needle_depth=0.5,seed=29",
            f"name=niah_{niah_context_length}_d90,source=niah,adapter=niah,"
            f"context_length={niah_context_length},needle_depth=0.9,seed=47",
        )
    )


def _smoke_datasets() -> str:
    return ";".join(
        (
            "name=gsm8k,source=openai/gsm8k,config=main,split=test,"
            "adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,"
            "prompt_field=question,answer_field=answer",
            "name=hotpotqa,source=hotpotqa/hotpot_qa,config=distractor,"
            "split=validation,adapter=hotpotqa,prompt_field=question,"
            "answer_field=answer,id_field=id",
            "name=niah_1024_d50,source=niah,adapter=niah,"
            "context_length=1024,needle_depth=0.5,seed=29",
        )
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


def _base_args(
    *,
    models: str,
    datasets: str,
    methods: str,
    samples: int,
    budget_ratios: str,
    thetas: str,
    recent_windows: str,
    alphas: str,
    max_length: int,
    max_new_tokens: int,
    prefill_block_size: int,
) -> dict[str, Any]:
    return {
        "models": models,
        "datasets": datasets,
        "methods": methods,
        "budget-ratios": budget_ratios,
        "thetas": thetas,
        "recent-windows": recent_windows,
        "alphas": alphas,
        "max-chunk-tokens": 64,
        "min-chunk-tokens": 5,
        "dependency-top-k": 8,
        "prefill-block-size": prefill_block_size,
        "max-samples": samples,
        "max-length": max_length,
        "max-new-tokens": max_new_tokens,
        "device": "cuda",
        "dtype": "float16",
        "attention-mode": "last",
        "layer-weighting": "linear",
        "tier1-score-mode": "dependency",
        "decode-policy": "common_streaming",
        "prompt-serialization": "auto",
        "truncation-side": "right",
        "allow-level2-fallback": True,
        "attn-implementation": "eager",
        "seed": 42,
        "experiment-variant": "default",
        "max-vram-fraction": 0.90,
    }


def build_local_jobs(
    profile: str,
    *,
    models: str = DEFAULT_LOCAL_MODELS,
    datasets: str | None = None,
    methods: str | None = None,
    max_samples: int | None = None,
    max_length: int = 3584,
    max_new_tokens: int = 96,
    prefill_block_size: int = 16,
    niah_context_length: int = 3072,
    budget_ratios: str | None = None,
    thetas: str | None = None,
    recent_windows: str | None = None,
    alphas: str | None = None,
    protocol_manifest: str | None = None,
    manifest_partition: str = "final",
    use_frozen_manifest: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    selected_datasets = datasets or _main_datasets(niah_context_length)
    if use_frozen_manifest:
        if not protocol_manifest:
            raise ValueError("--use-frozen-manifest requires --protocol-manifest.")
        selected_datasets = _attach_frozen_partition(
            selected_datasets,
            manifest=protocol_manifest,
            partition=manifest_partition,
        )

    jobs: list[tuple[str, dict[str, Any]]] = []
    if profile == "smoke":
        smoke_datasets = _smoke_datasets()
        if use_frozen_manifest:
            smoke_datasets = _attach_frozen_partition(
                smoke_datasets,
                manifest=str(protocol_manifest),
                partition=manifest_partition,
            )
        jobs.append(
            (
                "smoke",
                _base_args(
                    models=models,
                    datasets=smoke_datasets,
                    methods=methods or DEFAULT_METHODS,
                    samples=max_samples or 1,
                    budget_ratios=budget_ratios or "0.5",
                    thetas=thetas or DEFAULT_THETAS,
                    recent_windows=recent_windows or DEFAULT_RECENT_WINDOWS,
                    alphas=alphas or DEFAULT_ALPHAS,
                    max_length=min(max_length, 1536),
                    max_new_tokens=min(max_new_tokens, 64),
                    prefill_block_size=min(prefill_block_size, 16),
                ),
            )
        )

    if profile in {"main", "all"}:
        jobs.append(
            (
                "main",
                _base_args(
                    models=models,
                    datasets=selected_datasets,
                    methods=methods or DEFAULT_METHODS,
                    samples=max_samples or 50,
                    budget_ratios=budget_ratios or DEFAULT_BUDGET_RATIOS,
                    thetas=thetas or DEFAULT_THETAS,
                    recent_windows=recent_windows or DEFAULT_RECENT_WINDOWS,
                    alphas=alphas or DEFAULT_ALPHAS,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    prefill_block_size=prefill_block_size,
                ),
            )
        )

    if profile in {"param_sweep", "all"}:
        args = _base_args(
            models=models,
            datasets=selected_datasets,
            methods=methods or "tdc_kv",
            samples=max_samples or 20,
            budget_ratios=budget_ratios or DEFAULT_PARAM_BUDGET_RATIOS,
            thetas=thetas or DEFAULT_PARAM_THETAS,
            recent_windows=recent_windows or DEFAULT_PARAM_RECENT_WINDOWS,
            alphas=alphas or DEFAULT_PARAM_ALPHAS,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            prefill_block_size=prefill_block_size,
        )
        args["experiment-variant"] = "parameter_sweep"
        jobs.append(("param_sweep", args))

    if profile in {"baselines", "all"}:
        jobs.append(
            (
                "baselines",
                _base_args(
                    models=models,
                    datasets=selected_datasets,
                    methods=methods or BASELINE_METHODS,
                    samples=max_samples or 50,
                    budget_ratios=budget_ratios or DEFAULT_PARAM_BUDGET_RATIOS,
                    thetas=thetas or DEFAULT_THETAS,
                    recent_windows=recent_windows or DEFAULT_RECENT_WINDOWS,
                    alphas=alphas or DEFAULT_ALPHAS,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    prefill_block_size=prefill_block_size,
                ),
            )
        )

    if not jobs:
        raise ValueError(f"Unsupported local paper profile `{profile}`.")
    return jobs


def expand_jobs(
    jobs: list[tuple[str, dict[str, Any]]],
    *,
    split_models: bool,
    split_datasets: bool,
    sample_shards: int,
) -> list[tuple[str, dict[str, Any]]]:
    expanded: list[tuple[str, dict[str, Any]]] = []
    for name, arguments in jobs:
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


def _load_revision_file(path: str | None) -> str:
    if not path:
        return ""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("models") or {}
    return ";".join(f"{model}={revision}" for model, revision in mapping.items())


def _printable(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "main", "param_sweep", "baselines", "all"),
        default="smoke",
    )
    parser.add_argument("--models", default=DEFAULT_LOCAL_MODELS)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=3584)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prefill-block-size", type=int, default=16)
    parser.add_argument("--niah-context-length", type=int, default=3072)
    parser.add_argument("--budget-ratios", default=None)
    parser.add_argument("--thetas", default=None)
    parser.add_argument("--recent-windows", default=None)
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--output-root", default="outputs/local_paper")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-output-dir", default=None)
    parser.add_argument("--model-revisions", default="")
    parser.add_argument("--model-revisions-file", default=None)
    parser.add_argument("--protocol-manifest", default="protocol/local_paper_dataset_manifest.json")
    parser.add_argument("--use-frozen-manifest", action="store_true")
    parser.add_argument("--manifest-partition", default="final")
    parser.add_argument("--freeze-manifest", action="store_true")
    parser.add_argument("--partitions", default="qualification=5,tuning=20,final=50")
    parser.add_argument("--require-qualified", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-driver", action="store_true")
    parser.add_argument("--require-model-preflight", action="store_true")
    parser.add_argument("--require-model-revision", action="store_true")
    parser.add_argument("--require-no-truncation", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.freeze_manifest:
        datasets = args.datasets or _main_datasets(args.niah_context_length)
        command = [
            args.python,
            str(ROOT / "scripts" / "freeze_dataset_manifests.py"),
            "--datasets",
            datasets,
            "--partitions",
            args.partitions,
            "--output",
            str(Path(args.protocol_manifest).resolve()),
        ]
        print(_printable(command), flush=True)
        if not args.dry_run:
            completed = subprocess.run(command, cwd=ROOT, check=False)
            raise SystemExit(completed.returncode)
        return

    if args.use_frozen_manifest and not args.dry_run and not Path(args.protocol_manifest).exists():
        raise FileNotFoundError(
            f"Frozen protocol manifest is missing: {args.protocol_manifest}. "
            "Run with --freeze-manifest first."
        )

    jobs = build_local_jobs(
        args.profile,
        models=args.models,
        datasets=args.datasets,
        methods=args.methods,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        prefill_block_size=args.prefill_block_size,
        niah_context_length=args.niah_context_length,
        budget_ratios=args.budget_ratios,
        thetas=args.thetas,
        recent_windows=args.recent_windows,
        alphas=args.alphas,
        protocol_manifest=str(Path(args.protocol_manifest).resolve()),
        manifest_partition=args.manifest_partition,
        use_frozen_manifest=args.use_frozen_manifest,
    )
    jobs = expand_jobs(
        jobs,
        split_models=args.split_models,
        split_datasets=args.split_datasets,
        sample_shards=args.sample_shards,
    )

    if args.model_revisions and args.model_revisions_file:
        raise ValueError("Use either --model-revisions or --model-revisions-file.")
    model_revisions = args.model_revisions or _load_revision_file(args.model_revisions_file)
    if model_revisions:
        for _, arguments in jobs:
            arguments["model-revisions"] = model_revisions
    if args.require_model_revision:
        revision_mapping = _parse_model_revisions(model_revisions)
        required_models = {
            model.strip()
            for _, arguments in jobs
            for model in str(arguments["models"]).split(",")
            if model.strip()
        }
        missing = sorted(required_models - revision_mapping.keys())
        if missing:
            raise ValueError(f"Missing model revision pins for: {missing}")

    strict_flags = {
        "require-qualified": args.require_qualified,
        "require-cuda": args.require_cuda,
        "require-driver": args.require_driver,
        "require-model-preflight": args.require_model_preflight,
        "require-model-revision": args.require_model_revision,
        "require-no-truncation": args.require_no_truncation,
        "require-frozen-manifest": args.use_frozen_manifest,
    }
    for _, arguments in jobs:
        for key, enabled in strict_flags.items():
            if enabled:
                arguments[key] = True

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "models": [value.strip() for value in args.models.split(",") if value.strip()],
        "jobs": [],
    }
    for job_name, arguments in jobs:
        output = output_root / f"{job_name}.json"
        checkpoint = output_root / f"{job_name}.checkpoint.json"
        if args.resume and output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            failed_runs = int(existing.get("summary", {}).get("failed_runs", 1) or 0)
            if failed_runs == 0:
                manifest["jobs"].append(
                    {
                        "name": job_name,
                        "output": str(output),
                        "checkpoint": str(checkpoint),
                        "status": "skipped_complete",
                    }
                )
                print(f"\n[{job_name}] already complete; skipping.", flush=True)
                continue
        command = _command(
            args.python,
            arguments,
            output=output,
            checkpoint=checkpoint,
            resume=args.resume,
        )
        print(f"\n[{job_name}] {_printable(command)}", flush=True)
        job_record = {
            "name": job_name,
            "output": str(output),
            "checkpoint": str(checkpoint),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        manifest["jobs"].append(job_record)
        (output_root / "suite_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        job_record["return_code"] = int(completed.returncode)
        job_record["status"] = "complete" if completed.returncode == 0 else "failed"
        (output_root / "suite_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    (output_root / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nSuite manifest: {output_root / 'suite_manifest.json'}", flush=True)

    if args.summarize and not args.dry_run:
        summary_dir = Path(args.summary_output_dir).resolve() if args.summary_output_dir else output_root / "artifacts"
        summary_command = [
            args.python,
            str(ROOT / "scripts" / "summarize_local_paper_results.py"),
            "--inputs",
            str(output_root / "*.json"),
            "--output-dir",
            str(summary_dir),
        ]
        print(f"\n[summary] {_printable(summary_command)}", flush=True)
        completed = subprocess.run(summary_command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
