"""End-to-end pipeline health checks for TDC-KV and baselines.

This script provides staged checks:
- Stage A: core trace-based and synthetic checks (no model downloads required).
- Stage B: optional real-model smoke check (Phi-3/Llama wrapper prefill+compress).
- Stage C: repeated benchmark run to verify stability + failure accounting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.pipeline import (  # noqa: E402
    TraceSample,
    load_trace_samples,
    run_baseline_policy,
    run_tdc_full_pipeline_policy,
    run_tdc_policy,
)
from scripts.run_experiments import main as run_experiments_main  # noqa: E402


def _assert_budget(result: Any, budget: int, sequence_length: int) -> None:
    kept = int(result.kept_indices.numel())
    removed = int(result.removed_indices.numel())
    if kept > budget:
        raise AssertionError(f"Budget violated: kept={kept} > budget={budget}")
    if kept + removed != sequence_length:
        raise AssertionError(
            f"Token conservation failed: kept+removed={kept+removed} != t={sequence_length}"
        )


def _stage_a_trace_checks(sample_path: Path) -> dict[str, Any]:
    samples = load_trace_samples(sample_path)
    if not samples:
        raise RuntimeError(f"No samples found in {sample_path}")

    report: dict[str, Any] = {"samples": len(samples), "methods": {}}
    methods = ["tdc_kv", "chunkkv", "snapkv", "h2o"]

    for method in methods:
        ok = 0
        for sample in samples:
            budget = int(sample.budget or max(1, sample.sequence_length // 2))
            if method == "tdc_kv":
                result, _, _ = run_tdc_policy(sample, budget=budget)
            else:
                result, _ = run_baseline_policy(sample, method=method, budget=budget)
            _assert_budget(result, budget, sample.sequence_length)
            ok += 1
        report["methods"][method] = {"ok_samples": ok}
    return report


def _stage_a_full_pipeline_checks(sample_path: Path) -> dict[str, Any]:
    samples = load_trace_samples(sample_path)
    compatible: list[TraceSample] = [
        s for s in samples if s.token_ids is not None and s.attention_obs is not None
    ]
    if not compatible:
        return {"skipped": True, "reason": "no full-pipeline-compatible samples"}

    ok = 0
    for sample in compatible:
        budget = int(sample.budget or max(1, sample.sequence_length // 2))
        result, _, _, _ = run_tdc_full_pipeline_policy(
            sample,
            budget=budget,
            punct_ids=sample.punct_ids,
        )
        _assert_budget(result, budget, sample.sequence_length)
        ok += 1
    return {"skipped": False, "ok_samples": ok}


def _stage_b_model_smoke(model_name: str, budget: int, prompt: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    if "phi" in model_name.lower():
        from src.models.modeling_phi3 import Phi3TDCKVModel

        model = Phi3TDCKVModel.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    else:
        from src.models.modeling_llama import LlamaTDCKVModel

        model = LlamaTDCKVModel.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    out = model.prefill_and_compress(
        prompt=prompt,
        budget=budget,
        method="tdc_kv",
        window_size=16,
        recent_window=16,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "model": model_name,
        "budget": budget,
        "kept_tokens": int(out.kept_indices.numel()),
        "removed_tokens": int(out.removed_indices.numel()),
        "elapsed_ms": elapsed_ms,
    }


def _stage_c_repeated_experiments(sample_path: Path, output_path: Path) -> dict[str, Any]:
    argv = [
        "run_experiments.py",
        "--benchmarks",
        "niah",
        "--methods",
        "tdc_kv,chunkkv,snapkv,h2o",
        "--trace-overrides",
        f"niah={sample_path}",
        "--budget",
        "10",
        "--repeats",
        "5",
        "--warmup-runs",
        "1",
        "--allow-level2-fallback",
        "--output",
        str(output_path),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        run_experiments_main()
    finally:
        sys.argv = old_argv

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    bench = payload["benchmarks"]["niah"]["results"]
    return {
        method: {
            "failed_runs": int(block.get("failed_runs", 0)),
            "failure_reasons": block.get("failure_reasons", {}),
            "avg_latency_ms": float(block["summary"].get("avg_latency_ms", 0.0)),
        }
        for method, block in bench.items()
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-path", type=str, default="tmp/sample_trace.jsonl")
    p.add_argument("--output", type=str, default="outputs/pipeline_health_report.json")
    p.add_argument("--run-model-smoke", action="store_true")
    p.add_argument("--model-name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--model-budget", type=int, default=512)
    p.add_argument("--model-prompt", type=str, default="Explain KV cache compression in 3 bullets.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "stage_a_trace": _stage_a_trace_checks(trace_path),
        "stage_a_full_pipeline": _stage_a_full_pipeline_checks(trace_path),
        "stage_c_repeated_experiments": _stage_c_repeated_experiments(
            trace_path, output_path.parent / "pipeline_health_experiments.json"
        ),
    }

    if args.run_model_smoke:
        report["stage_b_model_smoke"] = _stage_b_model_smoke(
            model_name=args.model_name,
            budget=args.model_budget,
            prompt=args.model_prompt,
        )
    else:
        report["stage_b_model_smoke"] = {
            "skipped": True,
            "reason": "pass --run-model-smoke to enable model download/inference",
        }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote pipeline health report to {output_path}")


if __name__ == "__main__":
    main()
