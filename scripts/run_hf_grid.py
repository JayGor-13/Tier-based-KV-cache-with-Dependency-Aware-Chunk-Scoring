"""Run TDC-KV over multiple HuggingFace models, datasets, and parameters."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hf_runner import parse_dataset_spec, run_hf_grid
from benchmarks.io_utils import write_json_atomic
from benchmarks.qualification import qualification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Run HF grid search for TDC-KV")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of HF model names")
    parser.add_argument(
        "--model-revisions",
        default="",
        help="Semicolon-separated model=revision pins",
    )
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated list of dataset specs (e.g. source=gsm8k,config=main,split=test)")
    parser.add_argument("--budgets", type=str, default="", help="Comma-separated list of absolute budgets")
    parser.add_argument("--budget-ratios", type=str, default="", help="Comma-separated list of budget ratios (0.0 to 1.0)")
    parser.add_argument("--thetas", type=str, default="0.3", help="Comma-separated list of theta values")
    parser.add_argument("--recent-windows", type=str, default="16", help="Comma-separated list of recent windows")
    parser.add_argument("--alphas", type=str, default="0.6", help="Comma-separated list of alpha values")
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=64,
        help="Maximum tokens in a semantic chunk",
    )
    parser.add_argument(
        "--min-budget-utilization",
        type=float,
        default=0.99,
        help="Minimum allowed kept-token utilization of the target budget",
    )
    parser.add_argument(
        "--max-budget-shortfall-tokens",
        type=int,
        default=1,
        help="Maximum allowed target-budget underfill in tokens",
    )
    parser.add_argument(
        "--dependency-top-k",
        type=int,
        default=8,
        help="Maximum outgoing dependency edges retained per chunk",
    )
    parser.add_argument(
        "--prefill-block-size",
        type=int,
        default=128,
        help="Queries processed per bounded-attention prefill block",
    )
    parser.add_argument(
        "--tier1-score-mode",
        choices=("dependency", "fused", "none"),
        default="dependency",
        help="Signal used for Tier-1 assignment; `none` disables Tier 1",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("last", "all"),
        default="last",
        help="Attention layers used for scoring",
    )
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--min-chunk-tokens", type=int, default=5)
    parser.add_argument(
        "--chunking-strategy",
        choices=("sentence", "fixed", "token"),
        default="sentence",
        help="Semantic chunks, fixed-width chunks, or token-level ablation",
    )
    parser.add_argument("--fixed-chunk-size", type=int, default=16)
    parser.add_argument(
        "--layer-weighting",
        choices=("linear", "uniform"),
        default="linear",
    )
    parser.add_argument(
        "--protect-sink",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--protect-recent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="tdc_kv",
        help=(
            "Comma-separated methods: fullkv,streamingllm,h2o,snapkv,"
            "chunkkv,tdc_kv"
        ),
    )
    
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples per dataset")
    parser.add_argument("--max-length", type=int, default=2048, help="Max total sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument(
        "--prompt-serialization",
        choices=("auto", "raw", "chat"),
        default="auto",
        help="Model prompt formatting; auto uses a tokenizer chat template when available",
    )
    parser.add_argument(
        "--truncation-side",
        choices=("left", "right"),
        default="right",
        help="Side removed only when --max-length is exceeded",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--dtype", type=str, default="auto", help="Torch dtype")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--orchestration-job-id",
        default=None,
        help="Immutable suite job request hash used for safe skip/resume checks",
    )
    parser.add_argument(
        "--decode-policy",
        choices=("common_streaming", "tdc_native"),
        default="common_streaming",
        help="Use a common decode policy for fair comparisons or native TDC metadata",
    )
    parser.add_argument(
        "--experiment-variant",
        default="default",
        help="Explicit paper-reporting variant label",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        help="Transformers attention backend; eager is required by most models for attention outputs",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-level2-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Tier-2 fallback so matched-budget runs remain strict",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print live model, sample, prefill, and method progress",
    )
    parser.add_argument(
        "--fullkv-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare HF FullKV generation with the unpruned custom-cache path",
    )
    parser.add_argument(
        "--require-fullkv-parity",
        action="store_true",
        help="Exit nonzero after saving results unless every parity sample matches",
    )
    parser.add_argument(
        "--parity-max-samples",
        type=int,
        default=None,
        help="Limit parity controls per model/dataset while qualifying the full grid",
    )
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help=(
            "Exit nonzero unless all runs succeed, parity passes, generations are "
            "non-empty, and matched-budget/runtime contracts hold"
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Include recorded CUDA availability in the qualification gate",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--sample-shard-count", type=int, default=1)
    parser.add_argument(
        "--require-model-preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-vram-fraction", type=float, default=0.90)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-frozen-manifest", action="store_true")
    parser.add_argument("--require-model-revision", action="store_true")
    parser.add_argument("--require-driver", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--require-exact-niah", action="store_true")
    parser.add_argument("--require-no-truncation", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Incremental checkpoint JSON path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume successful runs from --checkpoint",
    )
    
    parser.add_argument("--output", type=str, default="outputs/hf_grid_results.json", help="Output JSON path")
    return parser.parse_args()

def main():
    args = parse_args()
    
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    model_revisions = {}
    for value in args.model_revisions.split(";"):
        if not value.strip():
            continue
        model, separator, revision = value.partition("=")
        if not separator or not model.strip() or not revision.strip():
            raise ValueError("Model revision pins must use model=revision entries.")
        model_revisions[model.strip()] = revision.strip()
    datasets = [parse_dataset_spec(d.strip()) for d in args.datasets.split(";") if d.strip()]
    
    budgets = [int(b.strip()) for b in args.budgets.split(",") if b.strip()]
    budget_ratios = [float(b.strip()) for b in args.budget_ratios.split(",") if b.strip()]
    thetas = [float(t.strip()) for t in args.thetas.split(",") if t.strip()]
    recent_windows = [int(w.strip()) for w in args.recent_windows.split(",") if w.strip()]
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    output_path = Path(args.output)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    )
    
    print(
        f"Running grid search for {len(models)} models and "
        f"{len(datasets)} datasets...",
        flush=True,
    )
    
    results = run_hf_grid(
        model_names=models,
        model_revisions=model_revisions,
        dataset_specs=datasets,
        budgets=budgets,
        budget_ratios=budget_ratios,
        thetas=thetas,
        recent_windows=recent_windows,
        alphas=alphas,
        max_chunk_tokens=args.max_chunk_tokens,
        min_budget_utilization=args.min_budget_utilization,
        max_budget_shortfall_tokens=args.max_budget_shortfall_tokens,
        dependency_top_k=args.dependency_top_k,
        prefill_block_size=args.prefill_block_size,
        tier1_score_mode=args.tier1_score_mode,
        layer_weighting=args.layer_weighting,
        protect_sink=args.protect_sink,
        protect_recent=args.protect_recent,
        attention_mode=args.attention_mode,
        layer_index=args.layer_index,
        min_chunk_tokens=args.min_chunk_tokens,
        chunking_strategy=args.chunking_strategy,
        fixed_chunk_size=args.fixed_chunk_size,
        methods=methods,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=(
            None
            if args.attn_implementation.strip().lower() in {"none", "default"}
            else args.attn_implementation
        ),
        hf_token=os.getenv(args.hf_token_env) if args.hf_token_env else None,
        allow_level2_fallback=args.allow_level2_fallback,
        continue_on_error=args.continue_on_error,
        progress=args.progress,
        run_fullkv_parity=(
            args.fullkv_parity
            or args.require_fullkv_parity
            or args.require_qualified
        ),
        parity_max_samples=args.parity_max_samples,
        prompt_serialization=args.prompt_serialization,
        truncation_side=args.truncation_side,
        seed=args.seed,
        decode_policy=args.decode_policy,
        experiment_variant=args.experiment_variant,
        sample_shard_index=args.sample_shard_index,
        sample_shard_count=args.sample_shard_count,
        require_model_preflight=args.require_model_preflight,
        preflight_require_cuda=args.require_cuda,
        preflight_require_unquantized=(args.quantization == "none"),
        max_vram_fraction=args.max_vram_fraction,
        deterministic=args.deterministic,
        orchestration_job_id=args.orchestration_job_id,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strict_qualification_requested = any(
        (
            args.require_qualified,
            args.require_cuda,
            args.require_frozen_manifest,
            args.require_model_revision,
            args.require_driver,
            args.require_clean_git,
            args.require_model_preflight,
            args.require_exact_niah,
            args.require_no_truncation,
        )
    )
    if strict_qualification_requested:
        declared = results["summary"].get("qualification", {})
        compressed_methods = [
            method
            for method in results.get("grid", {}).get("methods", [])
            if method != "fullkv"
        ]
        results["summary"]["qualification"] = qualification_report(
            results,
            require_parity=args.require_qualified,
            require_cuda=args.require_cuda,
            require_frozen_manifest=args.require_frozen_manifest,
            require_model_revision=args.require_model_revision,
            require_driver=args.require_driver,
            require_clean_git=args.require_clean_git,
            require_preflight=args.require_model_preflight,
            require_exact_niah=args.require_exact_niah,
            require_no_truncation=args.require_no_truncation,
            require_fullkv_pairing=bool(compressed_methods),
            expected_parity_records=declared.get("expected_parity_records"),
            min_gsm8k_parse_rate=declared.get("min_gsm8k_parse_rate", 0.0),
        )
    write_json_atomic(output_path, results)
    print(f"Results saved to {output_path}", flush=True)
    if args.require_fullkv_parity:
        parity = results["summary"]["fullkv_parity"]
        if parity.get("all_passed") is not True:
            print(
                "FullKV parity requirement failed: "
                f"{parity.get('mismatched_samples', [])}",
                flush=True,
            )
            raise SystemExit(3)
    qualification = results["summary"].get("qualification", {})
    if strict_qualification_requested and not qualification.get(
        "passed", False
    ):
        print("Paper qualification failed:", flush=True)
        for failure in qualification.get("failures", []):
            print(f"- {failure}", flush=True)
        raise SystemExit(4)

if __name__ == "__main__":
    main()
