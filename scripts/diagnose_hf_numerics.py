"""Run the A-H FullKV dtype/prefill diagnostic matrix.

The matrix deliberately separates native generation, a whole-prompt eager
forward, and custom unpruned-cache generation. It is a diagnostic prerequisite,
not a paper benchmark: no compressed method is run here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hf_runner import (
    build_prompt_from_record,
    load_dataset_records,
    parse_dataset_spec,
)
from benchmarks.io_utils import write_json_atomic
from benchmarks.model_preflight import loaded_model_preflight
from benchmarks.numerical_validation import (
    NumericalIntegrityError,
    generation_health,
    require_finite_tensor,
    tensor_health,
    validate_token_ids,
)
from benchmarks.reproducibility import collect_environment_metadata, seed_everything
from src.models.cache_utils import (
    extract_full_kv_cache,
    generate_text,
    generate_text_with_evicted_cache,
    load_hf_model_and_tokenizer,
    model_device,
    prepare_prompt,
    run_hf_prefill,
)


DEFAULT_DATASET = (
    "name=gsm8k_diagnostic,source=openai/gsm8k,config=main,split=train,"
    "adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,"
    "prompt_field=question,answer_field=answer,"
    "manifest=protocol/gsm8k_diagnostic_manifest.json,partition=diagnostic"
)


def diagnostic_cases() -> list[dict[str, Any]]:
    """Return the stable case definitions used in reports and tests."""
    return [
        {"case": "A", "path": "native_fullkv", "dtype": "float16", "blocks": []},
        {"case": "B", "path": "direct_eager", "dtype": "float16", "blocks": []},
        {"case": "C", "path": "custom_fullkv", "dtype": "float16", "blocks": ["whole"]},
        {"case": "D", "path": "custom_fullkv", "dtype": "float16", "blocks": [128, 16]},
        {"case": "E", "path": "native_fullkv", "dtype": "bfloat16", "blocks": []},
        {"case": "F", "path": "direct_eager", "dtype": "bfloat16", "blocks": []},
        {"case": "G", "path": "custom_fullkv", "dtype": "bfloat16", "blocks": ["whole"]},
        {"case": "H", "path": "custom_fullkv", "dtype": "bfloat16", "blocks": [128, 16]},
    ]


def first_differing_token(left: Iterable[int], right: Iterable[int]) -> int | None:
    left_values = tuple(int(value) for value in left)
    right_values = tuple(int(value) for value in right)
    for index, (left_value, right_value) in enumerate(zip(left_values, right_values)):
        if left_value != right_value:
            return index
    if len(left_values) != len(right_values):
        return min(len(left_values), len(right_values))
    return None


def diagnostic_record_passes(record: dict[str, Any]) -> bool:
    if record.get("status") != "ok":
        return False
    health = record.get("generation_health")
    if isinstance(health, dict) and health.get("degenerate_repetition") is True:
        return False
    if record.get("path") == "custom_fullkv":
        return (
            record.get("native_prompt_token_match") is True
            and record.get("native_token_match") is True
            and record.get("direct_next_token_match") is True
        )
    return True


def _model_revision(bundle: Any) -> str | None:
    candidates = (
        getattr(getattr(bundle.model, "config", None), "_commit_hash", None),
        getattr(bundle.tokenizer, "init_kwargs", {}).get("_commit_hash"),
    )
    return next((str(value) for value in candidates if value), None)


def _prompt_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _direct_eager_forward(model: Any, prepared: Any, *, sample_id: str) -> dict[str, Any]:
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in prepared.model_inputs.items()}
    with torch.no_grad():
        outputs = model(
            **encoded,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
    if outputs.logits is None:
        raise RuntimeError("direct eager forward returned no logits")
    require_finite_tensor(
        "direct_logits", outputs.logits, stage="direct_eager", sample_id=sample_id
    )
    if not outputs.attentions or any(value is None for value in outputs.attentions):
        raise RuntimeError("direct eager forward did not return every attention layer")
    for layer, attention in enumerate(outputs.attentions):
        require_finite_tensor(
            "direct_attention",
            attention,
            stage="direct_eager",
            sample_id=sample_id,
            layer=layer,
        )
    k_cache, v_cache = extract_full_kv_cache(outputs.past_key_values, offload_to_cpu=True)
    require_finite_tensor("direct_k_cache", k_cache, stage="direct_eager", sample_id=sample_id)
    require_finite_tensor("direct_v_cache", v_cache, stage="direct_eager", sample_id=sample_id)
    final_logits = outputs.logits[0, -1].detach().cpu().clone()
    next_token_id = int(torch.argmax(final_logits).item())
    validate_token_ids(
        [next_token_id],
        vocab_size=getattr(getattr(model, "config", None), "vocab_size", None),
        name="direct_next_token_id",
        sample_id=sample_id,
    )
    return {
        "next_token_id": next_token_id,
        "next_token_logits": final_logits,
        "logit_health": tensor_health(final_logits).to_dict(),
        "attention_layers": len(outputs.attentions),
        "cache_shape": list(k_cache.shape),
    }


def _safe_failure(exc: BaseException) -> dict[str, Any]:
    failure = {
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "numerical_health": {"passed": False},
    }
    if isinstance(exc, NumericalIntegrityError):
        failure["numerical_failure"] = exc.to_dict()
    return failure


def _native_cases(
    *,
    model_name: str,
    revision: str | None,
    dtype: str,
    prompts: list[dict[str, str]],
    device: str,
    max_length: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], str | None]:
    case = "A" if dtype == "float16" else "E"
    records: list[dict[str, Any]] = []
    bundle = load_hf_model_and_tokenizer(
        model_name,
        revision=revision,
        device=device,
        dtype=dtype,
        attn_implementation=None,
    )
    resolved_revision = _model_revision(bundle)
    actual_backend = getattr(bundle.model.config, "_attn_implementation", None)
    try:
        for prompt_row in prompts:
            base = {
                "case": case,
                "path": "native_fullkv",
                "dtype": dtype,
                "sample_id": prompt_row["sample_id"],
                "prompt_sha256": _prompt_id(prompt_row["prompt"]),
                "attention_backend": actual_backend,
            }
            try:
                prepared = prepare_prompt(
                    tokenizer=bundle.tokenizer,
                    prompt=prompt_row["prompt"],
                    max_length=max_length,
                    serialization="raw",
                )
                generated = generate_text(
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    prompt=prompt_row["prompt"],
                    max_new_tokens=max_new_tokens,
                    return_details=True,
                    prepared_prompt=prepared,
                )
                records.append(
                    {
                        **base,
                        "status": "ok",
                        "prompt_token_ids": prepared.input_ids[0].tolist(),
                        "generated_token_ids": list(generated.token_ids),
                        "prediction": generated.text,
                        "generation_health": generation_health(
                            generated.token_ids,
                            max_new_tokens=max_new_tokens,
                            special_token_ids=tuple(
                                token_id
                                for token_id in (
                                    bundle.tokenizer.eos_token_id,
                                    bundle.tokenizer.pad_token_id,
                                )
                                if token_id is not None
                            ),
                        ),
                        "numerical_health": {"passed": True},
                    }
                )
            except Exception as exc:
                records.append({**base, **_safe_failure(exc)})
    finally:
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records, resolved_revision


def _eager_cases(
    *,
    model_name: str,
    revision: str | None,
    dtype: str,
    prompts: list[dict[str, str]],
    native_records: list[dict[str, Any]],
    device: str,
    max_length: int,
    max_new_tokens: int,
    block_sizes: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    direct_case, whole_case, block_case = (
        ("B", "C", "D") if dtype == "float16" else ("F", "G", "H")
    )
    records: list[dict[str, Any]] = []
    bundle = load_hf_model_and_tokenizer(
        model_name,
        revision=revision,
        device=device,
        dtype=dtype,
        attn_implementation="eager",
    )
    resolved_revision = _model_revision(bundle)
    preflight = loaded_model_preflight(
        model=bundle.model,
        device=bundle.device,
        required_context=max_length,
        prefill_block_size=max(block_sizes),
        require_cuda=str(device).lower() != "cpu",
        requested_dtype=dtype,
        requested_attention_backend="eager",
        require_unquantized=True,
    )
    native_by_sample = {
        str(row["sample_id"]): row
        for row in native_records
        if row.get("status") == "ok"
    }
    try:
        for prompt_row in prompts:
            sample_id = prompt_row["sample_id"]
            prompt = prompt_row["prompt"]
            prepared = prepare_prompt(
                tokenizer=bundle.tokenizer,
                prompt=prompt,
                max_length=max_length,
                serialization="raw",
            )
            common = {
                "dtype": dtype,
                "sample_id": sample_id,
                "prompt_sha256": _prompt_id(prompt),
                "prompt_token_ids": prepared.input_ids[0].tolist(),
                "attention_backend": getattr(
                    bundle.model.config, "_attn_implementation", None
                ),
            }
            try:
                direct = _direct_eager_forward(
                    bundle.model, prepared, sample_id=sample_id
                )
                direct_logits = direct.pop("next_token_logits")
                records.append(
                    {
                        **common,
                        "case": direct_case,
                        "path": "direct_eager",
                        "status": "ok",
                        **direct,
                        "numerical_health": {"passed": True},
                    }
                )
            except Exception as exc:
                direct_logits = None
                records.append(
                    {
                        **common,
                        "case": direct_case,
                        "path": "direct_eager",
                        **_safe_failure(exc),
                    }
                )

            variants = [(whole_case, int(prepared.input_ids.shape[1]), "whole")]
            variants.extend((block_case, size, str(size)) for size in block_sizes)
            for case, block_size, block_label in variants:
                base = {
                    **common,
                    "case": case,
                    "path": "custom_fullkv",
                    "prefill_block_size": block_label,
                }
                try:
                    prefill = run_hf_prefill(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        prompt=prompt,
                        sample_id=sample_id,
                        observation_window=min(64, int(prepared.input_ids.shape[1])),
                        attention_mode="all",
                        layer_weighting="linear",
                        dependency_top_k=8,
                        prefill_block_size=block_size,
                        prepared_prompt=prepared,
                    )
                    generated = generate_text_with_evicted_cache(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        first_new_token_id=int(prefill.next_token_id),
                        max_new_tokens=max_new_tokens,
                        k_cache=prefill.k_cache,
                        v_cache=prefill.v_cache,
                        original_sequence_length=prefill.sequence_length,
                        return_details=True,
                    )
                    native = native_by_sample.get(sample_id)
                    native_tokens = tuple(native.get("generated_token_ids", ())) if native else ()
                    custom_tokens = tuple(generated.token_ids)
                    max_logit_difference = None
                    if direct_logits is not None and prefill.next_token_logits is not None:
                        max_logit_difference = float(
                            torch.max(
                                torch.abs(
                                    direct_logits.float()
                                    - prefill.next_token_logits.float()
                                )
                            ).item()
                        )
                    records.append(
                        {
                            **base,
                            "status": "ok",
                            "prefill_blocks": prefill.prefill_blocks,
                            "next_token_id": prefill.next_token_id,
                            "direct_next_token_match": (
                                direct_logits is not None
                                and int(torch.argmax(direct_logits).item())
                                == prefill.next_token_id
                            ),
                            "max_logit_abs_difference": max_logit_difference,
                            "generated_token_ids": list(custom_tokens),
                            "prediction": generated.text,
                            "native_prompt_token_match": bool(
                                native
                                and native.get("prompt_token_ids")
                                == common["prompt_token_ids"]
                            ),
                            "native_token_match": bool(native and native_tokens == custom_tokens),
                            "first_native_token_difference": (
                                first_differing_token(native_tokens, custom_tokens)
                                if native
                                else None
                            ),
                            "generation_health": generation_health(
                                custom_tokens,
                                max_new_tokens=max_new_tokens,
                                special_token_ids=tuple(
                                    token_id
                                    for token_id in (
                                        bundle.tokenizer.eos_token_id,
                                        bundle.tokenizer.pad_token_id,
                                    )
                                    if token_id is not None
                                ),
                            ),
                            "numerical_health": {"passed": True},
                        }
                    )
                except Exception as exc:
                    records.append({**base, **_safe_failure(exc)})
    finally:
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records, preflight, resolved_revision


def run_diagnostic_matrix(
    *,
    model_name: str,
    revision: str | None,
    prompts: list[dict[str, str]],
    device: str,
    max_length: int,
    max_new_tokens: int,
    block_sizes: tuple[int, ...] = (128, 16),
    seed: int = 42,
) -> dict[str, Any]:
    seed_everything(seed)
    all_records: list[dict[str, Any]] = []
    preflights: dict[str, Any] = {}
    revisions: dict[str, str | None] = {}
    for dtype in ("float16", "bfloat16"):
        try:
            native_records, native_revision = _native_cases(
                model_name=model_name,
                revision=revision,
                dtype=dtype,
                prompts=prompts,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            native_case = "A" if dtype == "float16" else "E"
            native_records = [
                {
                    "case": native_case,
                    "path": "native_fullkv",
                    "dtype": dtype,
                    "sample_id": prompt["sample_id"],
                    **_safe_failure(exc),
                }
                for prompt in prompts
            ]
            native_revision = None
        all_records.extend(native_records)
        try:
            eager_records, preflight, eager_revision = _eager_cases(
                model_name=model_name,
                revision=revision,
                dtype=dtype,
                prompts=prompts,
                native_records=native_records,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                block_sizes=block_sizes,
            )
        except Exception as exc:
            case_ids = ("B", "C", "D") if dtype == "float16" else ("F", "G", "H")
            eager_records = [
                {
                    "case": case,
                    "path": "eager_matrix_setup",
                    "dtype": dtype,
                    "sample_id": prompt["sample_id"],
                    **_safe_failure(exc),
                }
                for prompt in prompts
                for case in case_ids
            ]
            preflight = {"passed": False, "issues": [f"{type(exc).__name__}: {exc}"]}
            eager_revision = None
        all_records.extend(eager_records)
        preflights[dtype] = preflight
        revisions[dtype] = eager_revision or native_revision

    case_summary = {}
    for case in "ABCDEFGH":
        rows = [record for record in all_records if record.get("case") == case]
        failed_checks = [
            {
                "sample_id": row.get("sample_id"),
                "prefill_block_size": row.get("prefill_block_size"),
                "error": row.get("error"),
                "native_prompt_token_match": row.get("native_prompt_token_match"),
                "native_token_match": row.get("native_token_match"),
                "direct_next_token_match": row.get("direct_next_token_match"),
            }
            for row in rows
            if not diagnostic_record_passes(row)
        ]
        case_summary[case] = {
            "records": len(rows),
            "passed": bool(rows) and not failed_checks,
            "errors": [row.get("error") for row in rows if row.get("status") != "ok"],
            "failed_checks": failed_checks,
        }
    revision_matches = {
        dtype: revision is None or resolved == revision
        for dtype, resolved in revisions.items()
    }
    return {
        "diagnostic_schema_version": 1,
        "model": model_name,
        "requested_revision": revision,
        "resolved_revisions": revisions,
        "revision_matches": revision_matches,
        "protocol": {
            "cases": diagnostic_cases(),
            "device": device,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "block_sizes": list(block_sizes),
            "greedy": True,
            "seed": seed,
        },
        "environment": collect_environment_metadata(seed=seed),
        "model_preflights": preflights,
        "case_summary": case_summary,
        "all_bf16_cases_passed": bool(preflights.get("bfloat16", {}).get("passed"))
        and revision_matches.get("bfloat16", False)
        and all(case_summary[case]["passed"] for case in "EFGH"),
        "records": all_records,
    }


def _load_prompts(dataset_spec: str, max_samples: int) -> list[dict[str, str]]:
    spec = parse_dataset_spec(dataset_spec)
    records = load_dataset_records(spec, max_samples=max_samples)
    prompts = []
    for index, record in enumerate(records):
        prompt, _ = build_prompt_from_record(record, spec)
        raw_sample_id = record.get(spec.id_field) if spec.id_field else None
        if raw_sample_id is None:
            raw_sample_id = record.get("__tdc_source_index", index)
        sample_id = str(raw_sample_id)
        prompts.append({"sample_id": sample_id, "prompt": prompt})
    if not prompts:
        raise ValueError("diagnostic dataset selected no prompts")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-sizes", default="128,16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/diagnostics/hf_numerical_matrix.json")
    options = parser.parse_args()
    block_sizes = tuple(int(value) for value in options.block_sizes.split(",") if value)
    if not block_sizes or any(value <= 0 for value in block_sizes):
        parser.error("--block-sizes must contain positive integers")
    prompts = _load_prompts(options.dataset, options.max_samples)
    report = run_diagnostic_matrix(
        model_name=options.model,
        revision=options.revision,
        prompts=prompts,
        device=options.device,
        max_length=options.max_length,
        max_new_tokens=options.max_new_tokens,
        block_sizes=block_sizes,
        seed=options.seed,
    )
    write_json_atomic(options.output, report)
    print(f"Wrote {Path(options.output).resolve()}")
    if not report["all_bf16_cases_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
