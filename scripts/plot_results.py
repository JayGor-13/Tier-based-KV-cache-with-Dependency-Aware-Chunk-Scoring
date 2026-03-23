"""Generate comparison plots from experiment result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc
    return plt


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bar_plot(
    *,
    plt: Any,
    x_labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig_w = max(8.0, 1.0 + 1.4 * len(x_labels))
    fig_h = 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n_series = max(len(series), 1)
    width = 0.8 / n_series
    x = list(range(len(x_labels)))

    for idx, (name, values) in enumerate(series.items()):
        offsets = [xi - 0.4 + width * idx + width / 2.0 for xi in x]
        ax.bar(offsets, values, width=width, label=name)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if len(series) > 1:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_from_experiments_json(
    *,
    plt: Any,
    payload: dict[str, Any],
    output_dir: Path,
) -> None:
    benchmarks = payload.get("benchmarks", {})
    if not benchmarks:
        return

    metric_keys = [
        ("avg_compression_ratio", "Avg Compression Ratio"),
        ("avg_retention_ratio", "Avg Retention Ratio"),
        ("avg_latency_ms", "Avg Latency (ms)"),
        ("avg_budget_gap", "Avg Budget Gap"),
    ]

    bench_names = list(benchmarks.keys())
    method_names: set[str] = set()
    for bench in bench_names:
        method_names.update(benchmarks[bench].get("results", {}).keys())
    methods = sorted(method_names)

    for metric_key, metric_label in metric_keys:
        series: dict[str, list[float]] = {}
        for method in methods:
            vals = []
            for bench in bench_names:
                summary = (
                    benchmarks[bench]
                    .get("results", {})
                    .get(method, {})
                    .get("summary", {})
                )
                vals.append(float(summary.get(metric_key, 0.0)))
            series[method] = vals

        _bar_plot(
            plt=plt,
            x_labels=bench_names,
            series=series,
            title=f"Method Comparison by Benchmark: {metric_label}",
            ylabel=metric_label,
            output_path=output_dir / f"benchmarks_{metric_key}.png",
        )

    aggregate = payload.get("aggregate_summary", {})
    if aggregate:
        agg_methods = sorted(aggregate.keys())
        for metric_key, metric_label in metric_keys:
            _bar_plot(
                plt=plt,
                x_labels=agg_methods,
                series={
                    metric_label: [
                        float(aggregate[m].get(metric_key, 0.0)) for m in agg_methods
                    ]
                },
                title=f"Aggregate {metric_label} by Method",
                ylabel=metric_label,
                output_path=output_dir / f"aggregate_{metric_key}.png",
            )


def _plot_from_main_and_baselines(
    *,
    plt: Any,
    main_payload: dict[str, Any] | None,
    baseline_payload: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    if main_payload is None and baseline_payload is None:
        return

    combined: dict[str, dict[str, float]] = {}
    if main_payload is not None:
        combined["tdc_kv"] = {
            k: float(v) for k, v in main_payload.get("summary", {}).items() if k.startswith("avg_")
        }
    if baseline_payload is not None:
        for method, block in baseline_payload.get("results", {}).items():
            combined[method] = {
                k: float(v)
                for k, v in block.get("summary", {}).items()
                if k.startswith("avg_")
            }

    if not combined:
        return

    methods = sorted(combined.keys())
    metrics = sorted({m for vals in combined.values() for m in vals.keys()})
    for metric in metrics:
        _bar_plot(
            plt=plt,
            x_labels=methods,
            series={metric: [combined[m].get(metric, 0.0) for m in methods]},
            title=f"{metric} by Method",
            ylabel=metric,
            output_path=output_dir / f"summary_{metric}.png",
        )


def _line_plot_latency_sweep(
    *,
    plt: Any,
    payload: dict[str, Any],
    output_dir: Path,
) -> None:
    rows = payload.get("rows", [])
    if not rows:
        return
    methods = sorted({str(row.get("method", "")) for row in rows if row.get("method")})
    lengths = sorted(
        {int(row.get("sequence_length", 0)) for row in rows if int(row.get("sequence_length", 0)) > 0}
    )
    if not methods or not lengths:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        y_vals = []
        for length in lengths:
            match = next(
                (
                    row
                    for row in rows
                    if str(row.get("method", "")) == method
                    and int(row.get("sequence_length", 0)) == length
                ),
                None,
            )
            y_vals.append(float(match.get("avg_latency_ms", 0.0)) if match else 0.0)
        ax.plot(lengths, y_vals, marker="o", label=method)
    ax.set_title("Latency vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Average Latency (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_vs_sequence_length.png", dpi=180)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-json",
        type=str,
        default=None,
        help="Output JSON from scripts/run_experiments.py",
    )
    parser.add_argument(
        "--main-json",
        type=str,
        default="tmp/main_results.json",
        help="Output JSON from scripts/run_main_results.py",
    )
    parser.add_argument(
        "--baselines-json",
        type=str,
        default="tmp/baselines_results.json",
        help="Output JSON from scripts/run_baselines.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/plots",
        help="Directory to write PNG plots.",
    )
    parser.add_argument(
        "--latency-sweep-json",
        type=str,
        default=None,
        help="Output JSON from scripts/run_latency_sweep.py",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    plt = _require_matplotlib()
    output_dir = _ensure_dir(args.output_dir)

    if args.experiments_json:
        experiments_path = Path(args.experiments_json)
        if not experiments_path.exists():
            raise FileNotFoundError(f"Experiments JSON not found: {experiments_path}")
        exp_payload = _read_json(experiments_path)
        _plot_from_experiments_json(plt=plt, payload=exp_payload, output_dir=output_dir)

    main_payload = None
    baselines_payload = None
    main_path = Path(args.main_json)
    if main_path.exists():
        main_payload = _read_json(main_path)
    base_path = Path(args.baselines_json)
    if base_path.exists():
        baselines_payload = _read_json(base_path)
    _plot_from_main_and_baselines(
        plt=plt,
        main_payload=main_payload,
        baseline_payload=baselines_payload,
        output_dir=output_dir,
    )

    if args.latency_sweep_json:
        sweep_path = Path(args.latency_sweep_json)
        if not sweep_path.exists():
            raise FileNotFoundError(f"Latency sweep JSON not found: {sweep_path}")
        sweep_payload = _read_json(sweep_path)
        _line_plot_latency_sweep(plt=plt, payload=sweep_payload, output_dir=output_dir)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
