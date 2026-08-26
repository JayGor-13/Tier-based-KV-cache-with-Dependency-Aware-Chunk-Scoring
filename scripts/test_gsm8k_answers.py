"""Verify that an uncompressed Hugging Face model can answer GSM8K.

This is intentionally independent of the repository's KV-cache prefill,
scoring, eviction, and decode code.  Run it before any compression experiment.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import judge_gsm8k_prediction
from benchmarks.hf_runner import (
    build_prompt_from_record,
    load_dataset_records,
    parse_dataset_spec,
)
from benchmarks.io_utils import write_json_atomic
from benchmarks.reproducibility import seed_everything
from benchmarks.runtime_metrics import measure_call
from src.models.cache_utils import (
    generate_text,
    load_hf_model_and_tokenizer,
    prepare_prompt,
    resolve_greedy_generation_policy,
)


PROMPT_PROTOCOLS = {
    "direct": "default",
    "chunkkv8": "chunkkv_gsm8k_8shot",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test native, uncompressed GSM8K answers before running any "
            "KV-cache experiment."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face causal language model",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--prompt",
        choices=tuple(PROMPT_PROTOCOLS),
        default="direct",
        help="direct zero-shot prompt or the repository's ChunkKV eight-shot prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--serialization",
        choices=("raw", "auto", "chat"),
        default="raw",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        help="auto preserves the checkpoint dtype; fp16, bf16, and fp32 are accepted",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="attention backend; eager matches the compression runner",
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
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail before generation unless the loaded model uses CUDA",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-nonzero-accuracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exit nonzero after saving results when no tested answer is correct",
    )
    parser.add_argument(
        "--output",
        default="outputs/gsm8k/native_answers.json",
    )
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
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


def _dominant_parameter_dtype(model: Any) -> str | None:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype).replace("torch.", "")
        counts[name] = counts.get(name, 0) + int(parameter.numel())
    return max(counts, key=counts.get) if counts else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    nonempty = sum(bool(str(row.get("prediction", "")).strip()) for row in records)
    parseable = sum(bool(row.get("parseable")) for row in records)
    correct = sum(bool(row.get("correct")) for row in records)
    generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in records)
    elapsed_ms = sum(float(row.get("latency_ms", 0.0)) for row in records)
    return {
        "samples": count,
        "nonempty_answers": nonempty,
        "parseable_answers": parseable,
        "correct_answers": correct,
        "nonempty_rate": nonempty / count if count else 0.0,
        "parse_rate": parseable / count if count else 0.0,
        "accuracy": correct / count if count else 0.0,
        "generated_tokens": generated_tokens,
        "total_generation_ms": elapsed_ms,
        "mean_generation_ms": elapsed_ms / count if count else None,
        "generation_tokens_per_second": (
            generated_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0.0 else None
        ),
        "truncated_prompts": sum(bool(row.get("prompt_truncated")) for row in records),
        "generation_limit_answers": sum(
            bool(row.get("reached_generation_limit")) for row in records
        ),
        "generation_limit_rate": (
            sum(bool(row.get("reached_generation_limit")) for row in records) / count
            if count
            else 0.0
        ),
    }


def _print_record(row: dict[str, Any]) -> None:
    label = "CORRECT" if row["correct"] else "WRONG"
    print(
        f"\n[{row['sample_number']}] {label} | "
        f"parsed={row['normalized_prediction']!r} "
        f"gold={row['normalized_gold']!r} "
        f"tokens={row['generated_tokens']} "
        f"latency={row['latency_ms']:.1f} ms",
        flush=True,
    )
    print(f"Question: {row['question']}", flush=True)
    print(f"Model output:\n{row['prediction']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set, but PyTorch reports no CUDA device.")
    seed_everything(args.seed, deterministic=True)
    spec = _dataset_spec(
        prompt=args.prompt,
        dataset_revision=args.dataset_revision,
    )
    records = load_dataset_records(spec, max_samples=args.samples)
    if not records:
        raise RuntimeError("GSM8K returned no test records.")

    print(
        f"Loading {args.model} for native uncompressed generation...",
        flush=True,
    )
    bundle = load_hf_model_and_tokenizer(
        args.model,
        revision=args.model_revision,
        token=os.getenv(args.hf_token_env) if args.hf_token_env else None,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=(
            None
            if args.attn_implementation.strip().lower() in {"none", "default"}
            else args.attn_implementation
        ),
        quantization=args.quantization,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
    )
    if args.require_cuda and bundle.device.type != "cuda":
        raise RuntimeError(
            "--require-cuda was set, but the loaded model is not on a CUDA device."
        )
    generation_policy = resolve_greedy_generation_policy(
        bundle.model,
        bundle.tokenizer,
    )

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prompt, gold = build_prompt_from_record(record, spec)
        if gold is None:
            raise RuntimeError(f"GSM8K sample {index} has no gold answer.")
        prepared = prepare_prompt(
            tokenizer=bundle.tokenizer,
            prompt=prompt,
            max_length=args.max_length,
            serialization=args.serialization,
            truncation_side="right",
        )
        measured = measure_call(
            lambda: generate_text(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                return_details=True,
                prepared_prompt=prepared,
            ),
            device=bundle.device,
        )
        generation = measured.value
        judgment = judge_gsm8k_prediction(
            generation.text,
            gold,
            protocol=PROMPT_PROTOCOLS[args.prompt],
        )
        row = {
            "sample_number": index,
            "source_index": record.get("__tdc_source_index", index - 1),
            "question": str(record.get("question", "")),
            "raw_prompt": prompt,
            "rendered_prompt": prepared.rendered_text,
            "prediction": generation.text,
            "gold": gold,
            "normalized_prediction": judgment["normalized_prediction"],
            "normalized_gold": judgment["normalized_gold"],
            "parseable": bool(judgment["normalized_prediction"]),
            "correct": judgment["correct"],
            "score": judgment["score"],
            "generated_tokens": len(generation.token_ids),
            "reached_generation_limit": (
                len(generation.token_ids) >= args.max_new_tokens
            ),
            "generated_token_ids": list(generation.token_ids),
            "latency_ms": measured.measurement.elapsed_ms,
            "prompt_tokens": int(prepared.input_ids.shape[1]),
            "prompt_original_tokens": prepared.original_token_count,
            "prompt_truncated": prepared.was_truncated,
            "prompt_serialization": prepared.serialization,
        }
        results.append(row)
        _print_record(row)

    summary = summarize_records(results)
    payload = {
        "schema": "gsm8k_native_smoke_v1",
        "purpose": "native_uncompressed_answer_validation",
        "model": args.model,
        "requested_model_revision": args.model_revision,
        "loaded_model_revision": getattr(bundle.model.config, "_commit_hash", None),
        "device": str(bundle.device),
        "requested_dtype": args.dtype,
        "actual_parameter_dtype": _dominant_parameter_dtype(bundle.model),
        "requested_attention_backend": args.attn_implementation,
        "actual_attention_backend": getattr(
            bundle.model.config, "_attn_implementation", None
        ),
        "model_loading": dict(bundle.load_metadata),
        "prompt": args.prompt,
        "protocol": PROMPT_PROTOCOLS[args.prompt],
        "dataset": spec.to_dict(),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "serialization": args.serialization,
            "do_sample": False,
            "policy": generation_policy.to_dict(),
            "seed": args.seed,
        },
        "summary": summary,
        "records": results,
    }
    write_json_atomic(args.output, payload)
    print(
        "\nSummary: "
        f"parseable={summary['parseable_answers']}/{summary['samples']} "
        f"correct={summary['correct_answers']}/{summary['samples']} "
        f"accuracy={summary['accuracy']:.3f} "
        f"tokens/s={summary['generation_tokens_per_second'] or 0.0:.2f}",
        flush=True,
    )
    print(f"Saved: {Path(args.output).resolve()}", flush=True)

    if summary["parseable_answers"] == 0:
        print("FAIL: the model produced no parseable numeric answer.", flush=True)
        return 2
    if args.require_nonzero_accuracy and summary["correct_answers"] == 0:
        print("FAIL: accuracy is zero; inspect the saved raw outputs.", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
