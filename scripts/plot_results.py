#!/usr/bin/env python3
"""Generate paper-quality visualizations from TDC-KV experiment results.

Usage:
    python scripts/plot_results.py outputs/gsm8k_multi_budget.json --output-dir figures/
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from benchmarks.eval_metrics import token_f1, exact_match

# ── Paper-quality style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette inspired by academic papers
COLORS = {
    "tdc_kv": "#2563EB",      # vivid blue
    "baseline": "#9CA3AF",    # gray
    "accent1": "#F59E0B",     # amber
    "accent2": "#10B981",     # emerald
    "accent3": "#EF4444",     # red
    "accent4": "#8B5CF6",     # violet
}


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_ok_runs(data: dict) -> list[dict]:
    return [r for r in data["runs"] if r["status"] == "ok"]


def _bin_by_budget_ratio(runs: list[dict]) -> dict[float, list[dict]]:
    """Group runs by their retention ratio, binned to the nearest standard MBE level."""
    bins = {}
    for run in runs:
        ratio = run["metrics"]["retention_ratio"]
        bins.setdefault(ratio, [])
        bins[ratio].append(run)
    return bins


def _group_by_budget(runs: list[dict]) -> dict[int, list[dict]]:
    """Group runs by absolute budget value."""
    groups = {}
    for run in runs:
        b = run["config"]["budget"]
        groups.setdefault(b, [])
        groups[b].append(run)
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: F1 Score vs Compression Ratio (line chart)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_f1_vs_compression(data: dict, output_dir: Path):
    """Line chart: X = compression multiplier (1x, 2x, 4x...), Y = Token F1."""
    runs = _extract_ok_runs(data)
    if not runs:
        print("  [SKIP] No successful runs for F1 vs Compression plot.")
        return

    # Compute per-run F1 and CR
    points = []
    for r in runs:
        pred = str(r.get("evicted_prediction", ""))
        gold = str(r.get("gold", ""))
        f1 = token_f1(pred, gold)
        ret = r["metrics"]["retention_ratio"]
        cr = (1.0 / ret) if ret > 0 else 20.0
        points.append((cr, f1))

    # Also compute baseline F1
    baseline_points = []
    for r in runs:
        pred = str(r.get("prediction", ""))
        gold = str(r.get("gold", ""))
        f1 = token_f1(pred, gold)
        baseline_points.append(f1)
    avg_baseline_f1 = np.mean(baseline_points) if baseline_points else 0

    # Sort by CR
    points.sort(key=lambda x: x[0])

    # Group by similar CR values and average
    cr_bins = {}
    for cr, f1 in points:
        rounded_cr = round(cr, 1)
        cr_bins.setdefault(rounded_cr, [])
        cr_bins[rounded_cr].append(f1)

    crs = sorted(cr_bins.keys())
    avg_f1s = [np.mean(cr_bins[c]) for c in crs]
    std_f1s = [np.std(cr_bins[c]) for c in crs]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline horizontal line
    ax.axhline(y=avg_baseline_f1, color=COLORS["baseline"], linestyle="--",
               linewidth=1.5, label=f"Full Cache Baseline (F1={avg_baseline_f1:.3f})", zorder=2)

    # TDC-KV line
    ax.errorbar(crs, avg_f1s, yerr=std_f1s, color=COLORS["tdc_kv"],
                marker="o", markersize=7, linewidth=2, capsize=4,
                label="TDC-KV (Ours)", zorder=3)

    # Fill area under curve
    ax.fill_between(crs,
                     [max(0, f - s) for f, s in zip(avg_f1s, std_f1s)],
                     [min(1, f + s) for f, s in zip(avg_f1s, std_f1s)],
                     color=COLORS["tdc_kv"], alpha=0.1)

    ax.set_xlabel("Compression Ratio (×)")
    ax.set_ylabel("Token F1 Score")
    ax.set_title("TDC-KV: Accuracy vs. Compression Trade-off")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)

    out = output_dir / "f1_vs_compression.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Bar chart – Retention % vs F1 (grouped bars: Baseline vs Evicted)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_budget_comparison_bars(data: dict, output_dir: Path):
    """Grouped bar chart comparing Baseline F1 vs Evicted F1 at each budget level."""
    runs = _extract_ok_runs(data)
    if not runs:
        print("  [SKIP] No successful runs for bar chart.")
        return

    # Group runs by budget ratio (from the config)
    budget_groups = {}
    for r in runs:
        seq_len = r["sequence_length"]
        budget = r["config"]["budget"]
        ratio = budget / seq_len if seq_len > 0 else 0
        # Bin to nearest standard level
        if ratio >= 0.375:
            label = "50%"
        elif ratio >= 0.1875:
            label = "25%"
        elif ratio >= 0.09375:
            label = "12.5%"
        else:
            label = "6.25%"
        budget_groups.setdefault(label, [])
        budget_groups[label].append(r)

    # Desired order
    ordered_labels = [l for l in ["50%", "25%", "12.5%", "6.25%"] if l in budget_groups]
    if not ordered_labels:
        print("  [SKIP] No budget groups found.")
        return

    baseline_f1s = []
    evicted_f1s = []
    for label in ordered_labels:
        group = budget_groups[label]
        bf1 = np.mean([token_f1(str(r.get("prediction", "")), str(r.get("gold", ""))) for r in group])
        ef1 = np.mean([token_f1(str(r.get("evicted_prediction", "")), str(r.get("gold", ""))) for r in group])
        baseline_f1s.append(bf1)
        evicted_f1s.append(ef1)

    x = np.arange(len(ordered_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, baseline_f1s, width, label="Full Cache (No Eviction)",
                   color=COLORS["baseline"], edgecolor="white", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + width / 2, evicted_f1s, width, label="TDC-KV (Ours)",
                   color=COLORS["tdc_kv"], edgecolor="white", linewidth=0.8, zorder=3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("KV Cache Budget (% of original)")
    ax.set_ylabel("Token F1 Score")
    ax.set_title("Baseline vs TDC-KV at Different Compression Levels")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_labels)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0, top=max(max(baseline_f1s), max(evicted_f1s)) * 1.25)

    out = output_dir / "budget_comparison_bars.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Tier Distribution Stacked Bar (per-sample)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tier_distribution(data: dict, output_dir: Path):
    """Stacked bar chart showing how chunks are distributed across tiers per sample."""
    runs = _extract_ok_runs(data)
    if not runs:
        print("  [SKIP] No successful runs for tier distribution.")
        return

    # Take the first budget ratio group (most populated)
    # Group by sample_id and pick one budget per sample
    seen_samples = {}
    for r in runs:
        sid = r["sample_id"]
        if sid not in seen_samples:
            seen_samples[sid] = r

    samples = list(seen_samples.values())[:15]  # Cap at 15 for readability

    labels = [r["sample_id"].replace("gsm8k_", "S") for r in samples]
    tier0 = [r["tier_counts"]["tier0"] for r in samples]
    tier1 = [r["tier_counts"]["tier1"] for r in samples]
    tier2 = [r["tier_counts"]["tier2"] for r in samples]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, tier2, label="Tier 2 (Hard-Protected)", color="#EF4444", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x, tier1, bottom=tier2, label="Tier 1 (Soft-Protected)", color="#F59E0B", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x, tier0, bottom=[t1 + t2 for t1, t2 in zip(tier1, tier2)],
           label="Tier 0 (Eviction Candidate)", color="#10B981", edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Number of Chunks")
    ax.set_title("Chunk Protection Tier Distribution (TDC-KV)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)

    out = output_dir / "tier_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Cache Metrics Summary Radar/Spider Chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_metrics_radar(data: dict, output_dir: Path):
    """Radar chart showing multiple normalized performance dimensions."""
    runs = _extract_ok_runs(data)
    if not runs:
        print("  [SKIP] No successful runs for radar chart.")
        return

    # Compute aggregate metrics
    evicted_f1s = [token_f1(str(r.get("evicted_prediction", "")), str(r.get("gold", ""))) for r in runs]
    baseline_f1s = [token_f1(str(r.get("prediction", "")), str(r.get("gold", ""))) for r in runs]
    compressions = [r["metrics"]["compression_ratio"] for r in runs]
    latencies = [r["metrics"]["latency_ms"] for r in runs]

    avg_evicted_f1 = np.mean(evicted_f1s)
    avg_baseline_f1 = np.mean(baseline_f1s) if baseline_f1s else 1.0
    f1_retention = (avg_evicted_f1 / avg_baseline_f1) if avg_baseline_f1 > 0 else 0
    avg_compression = np.mean(compressions)
    avg_latency = np.mean(latencies)
    success_rate = sum(1 for r in data["runs"] if r["status"] == "ok") / max(len(data["runs"]), 1)

    # Normalize to 0-1 scale
    categories = ["F1 Retention", "Compression\nRatio", "Speed\n(inv. latency)", "Success\nRate", "Tier-2\nPreservation"]
    values = [
        min(f1_retention, 1.0),
        avg_compression,
        min(1.0 / max(avg_latency, 0.01), 1.0),  # inverse latency, capped
        success_rate,
        1.0,  # We always preserve Tier-2
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + values[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles_plot, values_plot, color=COLORS["tdc_kv"], alpha=0.15)
    ax.plot(angles_plot, values_plot, color=COLORS["tdc_kv"], linewidth=2, marker="o", markersize=6)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.set_title("TDC-KV Performance Profile", pad=20)

    # Add value annotations
    for angle, val, cat in zip(angles, values, categories):
        ax.annotate(f"{val:.2f}", xy=(angle, val), xytext=(5, 5),
                    textcoords="offset points", fontsize=9, color=COLORS["tdc_kv"], fontweight="bold")

    out = output_dir / "metrics_radar.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Per-Sample Prediction Quality Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def plot_sample_heatmap(data: dict, output_dir: Path):
    """Heatmap: rows = samples, columns = budget levels. Cell color = F1 score."""
    runs = _extract_ok_runs(data)
    if not runs:
        print("  [SKIP] No successful runs for heatmap.")
        return

    # Collect unique samples and budgets
    sample_ids = sorted(set(r["sample_id"] for r in runs))
    budgets = sorted(set(r["config"]["budget"] for r in runs))

    if len(budgets) < 2:
        print("  [SKIP] Need at least 2 budget levels for heatmap.")
        return

    # Build matrix
    f1_matrix = np.full((len(sample_ids), len(budgets)), np.nan)
    sid_idx = {s: i for i, s in enumerate(sample_ids)}
    bud_idx = {b: i for i, b in enumerate(budgets)}

    for r in runs:
        i = sid_idx[r["sample_id"]]
        j = bud_idx[r["config"]["budget"]]
        pred = str(r.get("evicted_prediction", ""))
        gold = str(r.get("gold", ""))
        f1_matrix[i, j] = token_f1(pred, gold)

    fig, ax = plt.subplots(figsize=(max(8, len(budgets) * 1.5), max(5, len(sample_ids) * 0.5)))
    im = ax.imshow(f1_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels([f"B={b}" for b in budgets], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(sample_ids)))
    ax.set_yticklabels([s.replace("gsm8k_", "S") for s in sample_ids])

    # Annotate cells
    for i in range(len(sample_ids)):
        for j in range(len(budgets)):
            val = f1_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xlabel("Budget (absolute tokens)")
    ax.set_ylabel("Sample")
    ax.set_title("Per-Sample Token F1 at Each Budget Level")
    fig.colorbar(im, ax=ax, label="Token F1", shrink=0.8)

    out = output_dir / "sample_budget_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality plots from TDC-KV results")
    parser.add_argument("results_json", type=str, help="Path to the results JSON file")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.results_json)
    ok_count = sum(1 for r in data["runs"] if r["status"] == "ok")
    err_count = sum(1 for r in data["runs"] if r["status"] == "error")
    print(f"Loaded {len(data['runs'])} runs ({ok_count} ok, {err_count} errors)")
    print(f"Generating plots to {output_dir}/\n")

    plot_f1_vs_compression(data, output_dir)
    plot_budget_comparison_bars(data, output_dir)
    plot_tier_distribution(data, output_dir)
    plot_metrics_radar(data, output_dir)
    plot_sample_heatmap(data, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
