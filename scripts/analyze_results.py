#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Add project root to sys.path so we can import benchmarks
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.eval_metrics import summarize_qa  # noqa: E402


def analyze_results(json_path: str):
    path = Path(json_path)
    if not path.exists():
        print(f"Error: {path} not found.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "runs" not in data or "summary" not in data:
        print("Error: Invalid JSON format. Missing 'runs' or 'summary' keys.")
        sys.exit(1)

    # Group runs by budget_ratio
    runs_by_ratio = {}
    for run in data["runs"]:
        if run["status"] != "ok":
            continue
            
        ratio = run["metrics"]["retention_ratio"]
        # Group to nearest standard MBE bin for presentation: 50%, 25%, 12.5%, 6.25%
        if ratio >= 0.35: bin_name = "50.0%"
        elif ratio >= 0.18: bin_name = "25.0%"
        elif ratio >= 0.09: bin_name = "12.5%"
        else: bin_name = "6.25%"
            
        if bin_name not in runs_by_ratio:
            runs_by_ratio[bin_name] = []
        runs_by_ratio[bin_name].append(run)

    print("### Matched-Budget Evaluation (MBE) Results\n")
    print(
        "| Method (Retention %) | Compression CR | Primary Metric | "
        "Baseline | Evicted | Delta |"
    )
    print(
        "|----------------------|----------------|----------------|"
        "----------|---------|-------|"
    )

    baseline_summary = data["summary"].get("baseline_qa_summary", {})
    baseline_metric = baseline_summary.get("primary_metric") or "unavailable"
    baseline_score = float(baseline_summary.get("primary_score") or 0.0) * 100

    for bin_name in ["50.0%", "25.0%", "12.5%", "6.25%"]:
        if bin_name not in runs_by_ratio:
            continue
            
        bin_runs = runs_by_ratio[bin_name]
        
        # Calculate avg CR
        multipliers = []
        for r in bin_runs:
            if "compression_multiplier" in r["metrics"]:
                multipliers.append(r["metrics"]["compression_multiplier"])
            elif r["metrics"]["retention_ratio"] > 0:
                multipliers.append(1.0 / r["metrics"]["retention_ratio"])
            else:
                multipliers.append(float('inf'))
                
        # Filter out inf for the average calculation
        valid_multipliers = [m for m in multipliers if m != float('inf')]
        avg_cr = sum(valid_multipliers) / len(valid_multipliers) if valid_multipliers else float('inf')
        
        qa_rows = [
            {
                "prediction": str(run.get("evicted_prediction", "")),
                "gold": str(run.get("gold", "")),
                "dataset": str(run.get("dataset", "")),
            }
            for run in bin_runs
            if run.get("gold") is not None
        ]
        evicted_summary = summarize_qa(qa_rows)
        evicted_metric = evicted_summary.get("primary_metric") or "unavailable"
        if baseline_metric != evicted_metric:
            raise ValueError(
                "Baseline and evicted metric families do not match: "
                f"{baseline_metric!r} != {evicted_metric!r}."
            )
        evicted_score = float(evicted_summary.get("primary_score") or 0.0) * 100
        score_delta = evicted_score - baseline_score
        
        # Format the row
        row = (
            f"| TDC-KV ({bin_name}) | {avg_cr:.2f}x | {evicted_metric} | "
            f"{baseline_score:.2f}% | {evicted_score:.2f}% | "
            f"{score_delta:+.2f}% |"
        )
        print(row)

if __name__ == "__main__":
    if len(sys.path) < 2:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <path_to_results.json>")
        sys.exit(1)
    analyze_results(sys.argv[1])
