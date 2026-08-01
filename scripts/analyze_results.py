#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Add project root to sys.path so we can import benchmarks
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    print("| Method (Retention %) | Compression CR | Baseline F1 | Evicted F1 | F1 Drop |")
    print("|----------------------|----------------|-------------|------------|---------|")

    # If the global baseline isn't perfect due to some failed runs, we calculate it dynamically
    baseline_f1 = data["summary"].get("baseline_qa_summary", {}).get("f1", 0.0) * 100

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
        
        # We need to manually calculate the evicted F1 for this specific bin
        # The F1 calculation here is a rough average of the F1 scores inside the bin
        from benchmarks.eval_metrics import token_f1
        f1_scores = []
        for r in bin_runs:
            pred = str(r.get("evicted_prediction", ""))
            gold = str(r.get("gold", ""))
            f1_scores.append(token_f1(pred, gold))
            
        evicted_f1 = (sum(f1_scores) / len(f1_scores)) * 100 if f1_scores else 0.0
        f1_drop = evicted_f1 - baseline_f1
        
        # Format the row
        row = f"| TDC-KV ({bin_name}) | {avg_cr:.2f}x | {baseline_f1:.2f}% | {evicted_f1:.2f}% | {f1_drop:+.2f}% |"
        print(row)

if __name__ == "__main__":
    if len(sys.path) < 2:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <path_to_results.json>")
        sys.exit(1)
    analyze_results(sys.argv[1])
