"""Latency sweep on synthetic cache traces across sequence lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.pipeline import (
    TraceSample,
    run_baseline_policy,
    run_tdc_full_pipeline_policy,
)
from src.core.chunker import MIN_CHUNK_TOKENS, SentenceBoundaryChunkConstructor
from src.core.pipeline import build_tdc_kv_pipeline
from src.core.scorer import DualSignalScorer


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_methods(text: str) -> list[str]:
    methods = [x.strip().lower() for x in text.split(",") if x.strip()]
    aliases = {"our": "tdc_kv", "our_model": "tdc_kv", "tdc": "tdc_kv"}
    normalized = [aliases.get(m, m) for m in methods]
    valid = {"tdc_kv", "chunkkv", "snapkv", "h2o"}
    bad = [m for m in normalized if m not in valid]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}. Allowed: {sorted(valid)}")
    return normalized


def _make_attention_obs(
    *,
    num_heads: int,
    window_size: int,
    sequence_length: int,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    w = max(1, min(window_size, sequence_length))
    raw = torch.rand(num_heads, w, sequence_length, generator=g, dtype=torch.float32)
    denom = raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return raw / denom


def _make_token_ids(
    *,
    sequence_length: int,
    punct_id: int,
    chunk_stride: int,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    ids = [rng.randint(100, 20000) for _ in range(sequence_length)]
    for pos in range(chunk_stride - 1, sequence_length, chunk_stride):
        ids[pos] = punct_id
    return torch.tensor(ids, dtype=torch.long)


def _build_synthetic_sample(
    *,
    sequence_length: int,
    budget: int,
    punct_id: int,
    chunk_stride: int,
    num_heads: int,
    head_dim: int,
    window_size: int,
    alpha: float,
    beta: float,
    min_chunk_tokens: int,
    seed: int,
) -> TraceSample:
    token_ids = _make_token_ids(
        sequence_length=sequence_length,
        punct_id=punct_id,
        chunk_stride=chunk_stride,
        seed=seed,
    )
    attention_obs = _make_attention_obs(
        num_heads=num_heads,
        window_size=window_size,
        sequence_length=sequence_length,
        seed=seed + 7,
    )

    chunker = SentenceBoundaryChunkConstructor(
        tokenizer=None,
        punct_ids={punct_id},
        min_chunk_tokens=min_chunk_tokens,
    )
    chunks, _ = chunker.forward(token_ids)
    scorer = DualSignalScorer(
        alpha=alpha,
        beta=beta,
        window_size=window_size,
    )
    chunk_scores = scorer.forward(attention_obs, chunks)

    gk = torch.Generator()
    gv = torch.Generator()
    gk.manual_seed(seed + 17)
    gv.manual_seed(seed + 27)
    k_cache = torch.randn(num_heads, sequence_length, head_dim, generator=gk)
    v_cache = torch.randn(num_heads, sequence_length, head_dim, generator=gv)

    return TraceSample(
        sample_id=f"synthetic_t{sequence_length}",
        chunks=chunks,
        chunk_scores=chunk_scores,
        k_cache=k_cache,
        v_cache=v_cache,
        token_ids=token_ids,
        punct_ids={punct_id},
        budget=budget,
        attention_obs=attention_obs,
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return float(var ** 0.5)


def _maybe_plot_latency(payload: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = payload.get("rows", [])
    if not rows:
        return
    methods = sorted({row["method"] for row in rows})
    lengths = sorted({int(row["sequence_length"]) for row in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        y = []
        for length in lengths:
            match = next(
                (
                    row
                    for row in rows
                    if row["method"] == method and int(row["sequence_length"]) == length
                ),
                None,
            )
            y.append(float(match["avg_latency_ms"]) if match else 0.0)
        ax.plot(lengths, y, marker="o", label=method)
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
        "--sequence-lengths",
        type=str,
        default="512,2048,8192",
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="tdc_kv,chunkkv,snapkv,h2o",
        help="Comma-separated methods.",
    )
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--recent-window", type=int, default=16)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--min-chunk-tokens", type=int, default=MIN_CHUNK_TOKENS)
    parser.add_argument("--chunk-stride", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--heavy-hitter-ratio", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-level2-fallback", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/latency_sweep.json",
        help="Output JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    lengths = _parse_csv_ints(args.sequence_lengths)
    methods = _parse_csv_methods(args.methods)
    repeats = max(1, int(args.repeats))
    warmup_runs = max(0, min(int(args.warmup_runs), repeats - 1))

    rows: list[dict[str, Any]] = []
    for idx, seq_len in enumerate(lengths):
        if seq_len <= 0:
            raise ValueError(f"Invalid sequence length: {seq_len}")
        budget = max(1, min(seq_len, int(round(float(args.budget_ratio) * seq_len))))
        sample = _build_synthetic_sample(
            sequence_length=seq_len,
            budget=budget,
            punct_id=99,
            chunk_stride=max(2, int(args.chunk_stride)),
            num_heads=max(1, int(args.num_heads)),
            head_dim=max(1, int(args.head_dim)),
            window_size=max(1, int(args.window_size)),
            alpha=float(args.alpha),
            beta=float(args.beta),
            min_chunk_tokens=max(1, int(args.min_chunk_tokens)),
            seed=args.seed + 1000 * idx,
        )

        for method in methods:
            latency_runs: list[float] = []
            kept_tokens = 0
            removed_tokens = 0
            failed = False
            error_msg = ""
            shared_pipeline = None
            if method == "tdc_kv":
                shared_pipeline = build_tdc_kv_pipeline(
                    punct_ids={99},
                    min_chunk_tokens=args.min_chunk_tokens,
                    alpha=args.alpha,
                    beta=args.beta,
                    window_size=args.window_size,
                    theta=args.theta,
                    recent_window=args.recent_window,
                    allow_level2_fallback=args.allow_level2_fallback,
                    device=sample.k_cache.device,
                )
            for run_i in range(repeats):
                try:
                    t0 = time.perf_counter()
                    if method == "tdc_kv":
                        result, _, _, _ = run_tdc_full_pipeline_policy(
                            sample,
                            budget=budget,
                            theta=args.theta,
                            recent_window=args.recent_window,
                            punct_ids={99},
                            min_chunk_tokens=args.min_chunk_tokens,
                            alpha=args.alpha,
                            beta=args.beta,
                            window_size=args.window_size,
                            allow_level2_fallback=args.allow_level2_fallback,
                            pipeline=shared_pipeline,
                        )
                    else:
                        result, _ = run_baseline_policy(
                            sample,
                            method=method,
                            budget=budget,
                            recent_window=args.recent_window,
                            theta=args.theta,
                            heavy_hitter_ratio=args.heavy_hitter_ratio,
                        )
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    if run_i >= warmup_runs:
                        latency_runs.append(elapsed)
                        kept_tokens = int(result.kept_indices.numel())
                        removed_tokens = int(result.removed_indices.numel())
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    error_msg = str(exc)
                    break

            row = {
                "sequence_length": seq_len,
                "budget": budget,
                "method": method,
                "repeats": repeats,
                "warmup_runs": warmup_runs,
                "measured_runs": len(latency_runs),
                "avg_latency_ms": _mean(latency_runs),
                "std_latency_ms": _std(latency_runs),
                "kept_tokens": kept_tokens,
                "removed_tokens": removed_tokens,
                "retention_ratio": (kept_tokens / float(seq_len)) if seq_len > 0 else 1.0,
                "compression_ratio": (removed_tokens / float(seq_len)) if seq_len > 0 else 0.0,
                "failed": failed,
                "error": error_msg,
            }
            rows.append(row)

    by_length: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        lkey = str(row["sequence_length"])
        by_length.setdefault(lkey, {})
        by_length[lkey][row["method"]] = row

    payload = {
        "config": {
            "sequence_lengths": lengths,
            "methods": methods,
            "budget_ratio": args.budget_ratio,
            "window_size": args.window_size,
            "recent_window": args.recent_window,
            "theta": args.theta,
            "alpha": args.alpha,
            "beta": args.beta,
            "min_chunk_tokens": args.min_chunk_tokens,
            "chunk_stride": args.chunk_stride,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "heavy_hitter_ratio": args.heavy_hitter_ratio,
            "repeats": repeats,
            "warmup_runs": warmup_runs,
            "seed": args.seed,
            "allow_level2_fallback": args.allow_level2_fallback,
        },
        "results": by_length,
        "rows": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.plot:
        _maybe_plot_latency(payload, output_path.parent)
    print(f"Wrote latency sweep to {output_path}")


if __name__ == "__main__":
    main()
