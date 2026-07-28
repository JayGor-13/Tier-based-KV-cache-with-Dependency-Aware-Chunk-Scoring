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
    
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples per dataset")
    parser.add_argument("--max-length", type=int, default=2048, help="Max total sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--dtype", type=str, default="auto", help="Torch dtype")
    parser.add_argument("--allow-level2-fallback", action="store_true", help="Allow eviction of tier-2 chunks if budget is too strict")
    
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
    
    print(f"Running grid search for {len(models)} models and {len(datasets)} datasets...")
    
    results = run_hf_grid(
        model_names=models,
        dataset_specs=datasets,
        budgets=budgets,
        budget_ratios=budget_ratios,
        thetas=thetas,
        recent_windows=recent_windows,
        alphas=alphas,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        allow_level2_fallback=args.allow_level2_fallback,
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
