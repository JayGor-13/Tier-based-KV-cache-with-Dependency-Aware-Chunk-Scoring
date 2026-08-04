"""Run TDC-KV over multiple HuggingFace models, datasets, and parameters."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hf_runner import parse_dataset_spec, run_hf_grid

def parse_args():
    parser = argparse.ArgumentParser(description="Run HF grid search for TDC-KV")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of HF model names")
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
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--dtype", type=str, default="auto", help="Torch dtype")
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
    
    parser.add_argument("--output", type=str, default="outputs/hf_grid_results.json", help="Output JSON path")
    return parser.parse_args()

def main():
    args = parse_args()
    
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [parse_dataset_spec(d.strip()) for d in args.datasets.split(";") if d.strip()]
    
    budgets = [int(b.strip()) for b in args.budgets.split(",") if b.strip()]
    budget_ratios = [float(b.strip()) for b in args.budget_ratios.split(",") if b.strip()]
    thetas = [float(t.strip()) for t in args.thetas.split(",") if t.strip()]
    recent_windows = [int(w.strip()) for w in args.recent_windows.split(",") if w.strip()]
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    
    print(
        f"Running grid search for {len(models)} models and "
        f"{len(datasets)} datasets...",
        flush=True,
    )
    
    results = run_hf_grid(
        model_names=models,
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
        methods=methods,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        allow_level2_fallback=args.allow_level2_fallback,
        progress=args.progress,
        run_fullkv_parity=(args.fullkv_parity or args.require_fullkv_parity),
        prompt_serialization=args.prompt_serialization,
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
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

if __name__ == "__main__":
    main()
