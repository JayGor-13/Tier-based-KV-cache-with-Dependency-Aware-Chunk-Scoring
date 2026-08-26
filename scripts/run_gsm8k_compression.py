"""Run a small FullKV-versus-TDC-KV GSM8K experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hf_runner import parse_dataset_spec, run_hf_grid
from benchmarks.io_utils import write_json_atomic


PROMPT_PROTOCOLS = {
    "direct": "default",
    "chunkkv8": "chunkkv_gsm8k_8shot",
}


def _ratio_list(value: str) -> list[float]:
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not ratios or any(not 0.0 < ratio < 1.0 for ratio in ratios):
        raise argparse.ArgumentTypeError(
            "retention ratios must be comma-separated values strictly between 0 and 1"
        )
    return ratios


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare unpruned FullKV with TDC-KV on a few GSM8K samples."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--prompt", choices=tuple(PROMPT_PROTOCOLS), default="direct")
    parser.add_argument(
        "--retention-ratios",
        type=_ratio_list,
        default=_ratio_list("0.5,0.3"),
        help="fractions of the original prompt KV tokens to keep",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="attention backend; eager is required for attention-score collection",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "bnb-4bit"),
        default="none",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        choices=("nf4", "fp4"),
        default="nf4",
    )
    parser.add_argument(
        "--bnb-4bit-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--serialization",
        choices=("raw", "auto", "chat"),
        default="raw",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--native-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also verify the custom unpruned path against native HF generation",
    )
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=5,
        help="maximum native/custom parity controls; limits full-run overhead",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail unless CUDA loading and CUDA artifact qualification pass",
    )
    parser.add_argument("--max-vram-fraction", type=float, default=0.90)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="outputs/gsm8k/compression.json")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.parity_samples <= 0:
        parser.error("--parity-samples must be positive")
    if not 0.0 < args.max_vram_fraction <= 1.0:
        parser.error("--max-vram-fraction must be in (0, 1]")
    return args


def _dataset_spec(*, prompt: str, dataset_revision: str | None):
    fields = [
        "name=gsm8k",
        "source=openai/gsm8k",
        "config=main",
        "split=test",
        "adapter=gsm8k",
        f"protocol={PROMPT_PROTOCOLS[prompt]}",
        "prompt_field=question",
        "answer_field=answer",
    ]
    if dataset_revision:
        fields.append(f"revision={dataset_revision}")
    return parse_dataset_spec(",".join(fields))


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        if row.get("status") != "ok":
            continue
        config = row.get("config") or {}
        method = str(row.get("method", config.get("method", "unknown")))
        retention = None
        if config.get("budget_type") == "ratio":
            retention = float(config.get("budget_value"))
        grouped[(method, retention)].append(row)

    summary = []
    for (method, retention), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or 1.0)
    ):
        judgments = [row.get("judgment") or {} for row in rows]
        latencies = [
            float((row.get("runtime") or {}).get("total_measured_ms", 0.0))
            for row in rows
        ]
        throughputs = [
            float(value)
            for row in rows
            for value in [(row.get("runtime") or {}).get("decode_tokens_per_second")]
            if value is not None
        ]
        actual_retentions = [
            float(value)
            for row in rows
            for value in [(row.get("metrics") or {}).get("retention_ratio")]
            if value is not None
        ]
        count = len(rows)
        summary.append(
            {
                "method": method,
                "retention_ratio": retention,
                "samples": count,
                "mean_actual_retention": (
                    sum(actual_retentions) / len(actual_retentions)
                    if actual_retentions
                    else None
                ),
                "parse_rate": (
                    sum(bool(judge.get("normalized_prediction")) for judge in judgments)
                    / count
                ),
                "accuracy": sum(float(judge.get("score", 0.0)) for judge in judgments)
                / count,
                "mean_total_latency_ms": sum(latencies) / count,
                "mean_decode_tokens_per_second": (
                    sum(throughputs) / len(throughputs) if throughputs else None
                ),
            }
        )
    return summary


def summarize_paired_quality(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each compressed answer with its paired FullKV control."""
    successful = [row for row in runs if row.get("status") == "ok"]
    fullkv = {
        str(row.get("sample_id")): row
        for row in successful
        if row.get("method") == "fullkv"
    }
    compressed: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        if row.get("method") != "tdc_kv":
            continue
        config = row.get("config") or {}
        if config.get("budget_type") != "ratio":
            continue
        compressed[float(config["budget_value"])].append(row)

    summaries = []
    for requested_retention, rows in sorted(compressed.items()):
        pairs = [
            (fullkv[str(row.get("sample_id"))], row)
            for row in rows
            if str(row.get("sample_id")) in fullkv
        ]
        if not pairs:
            continue
        full_scores = [
            float((full.get("judgment") or {}).get("score", 0.0))
            for full, _compressed in pairs
        ]
        compressed_scores = [
            float((compressed_row.get("judgment") or {}).get("score", 0.0))
            for _full, compressed_row in pairs
        ]
        count = len(pairs)
        full_correct = sum(score == 1.0 for score in full_scores)
        compressed_correct = sum(score == 1.0 for score in compressed_scores)
        preserved = sum(
            full_score == 1.0 and compressed_score == 1.0
            for full_score, compressed_score in zip(
                full_scores,
                compressed_scores,
                strict=True,
            )
        )
        regressions = sum(
            full_score == 1.0 and compressed_score != 1.0
            for full_score, compressed_score in zip(
                full_scores,
                compressed_scores,
                strict=True,
            )
        )
        recoveries = sum(
            full_score != 1.0 and compressed_score == 1.0
            for full_score, compressed_score in zip(
                full_scores,
                compressed_scores,
                strict=True,
            )
        )
        full_accuracy = full_correct / count
        compressed_accuracy = compressed_correct / count
        summaries.append(
            {
                "retention_ratio": requested_retention,
                "paired_samples": count,
                "fullkv_accuracy": full_accuracy,
                "tdc_kv_accuracy": compressed_accuracy,
                "accuracy_delta": compressed_accuracy - full_accuracy,
                "accuracy_retention": (
                    compressed_accuracy / full_accuracy
                    if full_accuracy > 0.0
                    else None
                ),
                "fullkv_correct": full_correct,
                "tdc_kv_correct": compressed_correct,
                "preserved_fullkv_correct": preserved,
                "paired_correct_preservation": (
                    preserved / full_correct if full_correct else None
                ),
                "regressions": regressions,
                "recoveries": recoveries,
            }
        )
    return summaries


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "\nmethod       requested  actual  samples  parse   accuracy  "
        "latency_ms  decode_tok/s"
    )
    for row in rows:
        retention = "full" if row["retention_ratio"] is None else f"{row['retention_ratio']:.2f}"
        actual = row["mean_actual_retention"]
        actual_text = "n/a" if actual is None else f"{actual:.2f}"
        throughput = row["mean_decode_tokens_per_second"]
        print(
            f"{row['method']:<12} {retention:>9}  {actual_text:>6}  "
            f"{row['samples']:>7}  "
            f"{row['parse_rate']:>5.2f}   {row['accuracy']:>8.3f}  "
            f"{row['mean_total_latency_ms']:>10.1f}  "
            f"{throughput if throughput is not None else 0.0:>12.2f}"
        )


def _print_paired_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print(
        "\nrequested  pairs  full_em  tdc_em  delta    accuracy_retention  "
        "preserved  regressions  recoveries"
    )
    for row in rows:
        retention = row["accuracy_retention"]
        retention_text = "n/a" if retention is None else f"{retention:.3f}"
        preserved = row["paired_correct_preservation"]
        preserved_text = "n/a" if preserved is None else f"{preserved:.3f}"
        print(
            f"{row['retention_ratio']:>9.2f}  {row['paired_samples']:>5}  "
            f"{row['fullkv_accuracy']:>7.3f}  {row['tdc_kv_accuracy']:>6.3f}  "
            f"{row['accuracy_delta']:>+6.3f}  {retention_text:>18}  "
            f"{preserved_text:>9}  {row['regressions']:>11}  "
            f"{row['recoveries']:>10}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_path.with_suffix(".checkpoint.sqlite")
    )
    spec = _dataset_spec(
        prompt=args.prompt,
        dataset_revision=args.dataset_revision,
    )
    results = run_hf_grid(
        model_names=[args.model],
        model_revisions=(
            {args.model: args.model_revision} if args.model_revision else {}
        ),
        dataset_specs=[spec],
        budgets=[],
        budget_ratios=args.retention_ratios,
        thetas=[0.3],
        recent_windows=[16],
        alphas=[0.6],
        methods=["fullkv", "tdc_kv"],
        max_samples=args.samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        attention_mode="last",
        min_chunk_tokens=5,
        max_chunk_tokens=64,
        prefill_block_size=128,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=(
            None
            if args.attn_implementation.strip().lower() in {"none", "default"}
            else args.attn_implementation
        ),
        quantization=args.quantization,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
        hf_token=os.getenv(args.hf_token_env) if args.hf_token_env else None,
        continue_on_error=False,
        progress=True,
        run_fullkv_parity=args.native_parity,
        parity_max_samples=args.parity_samples if args.native_parity else None,
        prompt_serialization=args.serialization,
        truncation_side="right",
        seed=args.seed,
        decode_policy="common_streaming",
        experiment_variant="gsm8k_foundation",
        require_model_preflight=args.require_cuda,
        preflight_require_cuda=args.require_cuda,
        preflight_require_unquantized=(args.quantization == "none"),
        max_vram_fraction=args.max_vram_fraction,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )
    simple_summary = summarize_runs(results.get("runs", []))
    paired_summary = summarize_paired_quality(results.get("runs", []))
    results["gsm8k_foundation_summary"] = simple_summary
    results["gsm8k_paired_summary"] = paired_summary
    write_json_atomic(output_path, results)
    _print_summary(simple_summary)
    _print_paired_summary(paired_summary)
    print(f"\nSaved: {output_path.resolve()}")

    expected_groups = 1 + len(args.retention_ratios)
    if len(simple_summary) != expected_groups:
        print(
            f"FAIL: expected {expected_groups} result groups, got {len(simple_summary)}."
        )
        return 2
    parity = (results.get("summary") or {}).get("fullkv_parity") or {}
    if args.native_parity and parity.get("all_passed") is not True:
        print("FAIL: custom FullKV does not match native Hugging Face generation.")
        return 3
    qualification = (results.get("summary") or {}).get("qualification") or {}
    if args.require_cuda and qualification.get("passed") is not True:
        print("FAIL: CUDA qualification did not pass.")
        for failure in qualification.get("failures", []):
            print(f"- {failure}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
