# TDC-KV

Tier-based KV-cache eviction with dependency-aware chunk scoring.

This repository contains a trace-driven prototype for compressing transformer
KV caches while preserving sink tokens, recent tokens, and high-importance
chunks. The implementation is split into four core modules:

- `src/core/chunker.py`: sentence/punctuation-boundary chunk construction.
- `src/core/scorer.py`: dual-signal attention and forward-routing chunk scores.
- `src/core/masker.py`: Tier 0/1/2 protection assignment.
- `src/core/evictor.py`: priority-respecting KV-cache eviction.

Benchmark utilities live under `benchmarks/`, with command-line experiment
runners under `scripts/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

With conda:

```bash
conda env create -f environment.yml
conda activate tdc-kv
pip install -e ".[dev]"
```

## Verify

```bash
python -m pytest
python -c "import src.core; import src.baselines; print('imports ok')"
```

## Trace Smoke Runs

```bash
python scripts/run_main_results.py --trace-path data/sample_trace.jsonl --recent-window 4 --output outputs/main_results.json
python scripts/run_baselines.py --trace-path data/sample_trace.jsonl --recent-window 4 --output outputs/baselines_results.json
python scripts/run_ablations.py --trace-path data/sample_trace.jsonl --theta-grid 0.3 --recent-window-grid 4 --output outputs/ablations_results.json
```

Generated outputs are ignored by git.
