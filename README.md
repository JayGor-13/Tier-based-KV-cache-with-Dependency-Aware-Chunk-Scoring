# TDC-KV

Tier-based KV-cache eviction with dependency-aware chunk scoring.

This repository contains a trace-driven prototype for compressing transformer
KV caches while preserving sink tokens, recent tokens, and high-importance
chunks. The implementation is split into four core modules:

- `src/core/chunker.py`: sentence/punctuation-boundary chunk construction.
- `src/core/dependency_graph.py`: sparse historical chunk-dependency collection.
- `src/core/scorer.py`: attention-mass and graph-routed dependency scores.
- `src/core/masker.py`: Tier 0/1/2 protection assignment.
- `src/core/evictor.py`: priority-respecting KV-cache eviction.
- `src/models/cache_manager.py`: strict-budget re-eviction during decoding.

Live HuggingFace prefill is blockwise: each query block consumes its attention
rows into the sparse dependency graph and final observation window, then releases
the dense block attention tensors. The full prompt KV cache is retained until the
configured eviction policy is applied.

Benchmark utilities live under `benchmarks/`, with command-line experiment
runners under `scripts/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

On Colab, keep the runtime-provided PyTorch build. The text-only pipeline does
not require `torchvision` or `torchaudio`; remove those optional wheels if their
CUDA build differs from PyTorch. The bundled quickstart performs this cleanup
before importing Transformers.

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

The offline HuggingFace end-to-end tests instantiate tiny random GPT-2, Llama,
and Qwen2 models. They verify compressed-cache logit parity, global logical
position handling across Transformers 4/5, exact token-granular budgets, and
repeated decode-time re-eviction without downloading checkpoints.

## Trace Smoke Runs

```bash
python scripts/run_main_results.py --trace-path data/sample_trace.jsonl --recent-window 4 --output outputs/main_results.json
python scripts/run_baselines.py --trace-path data/sample_trace.jsonl --recent-window 4 --output outputs/baselines_results.json
python scripts/run_ablations.py --trace-path data/sample_trace.jsonl --theta-grid 0.3 --recent-window-grid 4 --output outputs/ablations_results.json
```

Generated outputs are ignored by git.

## HuggingFace Smoke Run

```bash
python scripts/run_hf_grid.py \
  --models Qwen/Qwen2.5-0.5B-Instruct \
  --datasets "name=gsm8k,source=openai/gsm8k,config=main,split=test,prompt_field=question,answer_field=answer" \
  --methods fullkv,tdc_kv \
  --budget-ratios 0.5 \
  --max-chunk-tokens 64 \
  --min-budget-utilization 0.99 \
  --max-budget-shortfall-tokens 1 \
  --allow-level2-fallback \
  --prefill-block-size 128 \
  --max-samples 5 \
  --output outputs/hf_smoke.json
```

`--max-chunk-tokens` bounds punctuation-free spans. Eviction removes ranked
whole chunks until the final boundary, then trims only the oldest required
positions from that chunk to satisfy the matched token budget. The utilization
and shortfall flags make this an executable result contract. Lower
`--prefill-block-size` to reduce peak attention memory. Larger blocks use
fewer model calls but materialize larger `[layers, heads, block, prefix]`
attention tensors. Result JSON records the configured size and actual block count.

Long HuggingFace runs print live model, sample, prefill, and method/budget
progress by default. Pass `--no-progress` only when quiet output is required.

For staged Colab execution, use `notebooks/tdc_kv_actual_testing.ipynb`. It
verifies `branch-h`, CUDA, focused end-to-end tests, and one-sample smoke runs
before exposing the larger GSM8K, NIAH, and HotpotQA pilot cells. Each run has
a timeout and writes to a unique timestamped output directory.

HF result JSON also contains `grouped_results`, aggregated by model, dataset,
method, requested budget specification, and all remaining configuration values.
Each group includes run/error counts, cache and QA metrics, sequence and resolved
budget distributions, decode-cache diagnostics, and tier-count summaries. QA
summaries expose a dataset-aware `primary_metric`/`primary_score`: extracted
numeric accuracy for GSM8K, exact needle retrieval accuracy for NIAH, and the
official normalized answer F1 for HotpotQA. Generic full-output EM/F1 remain as
diagnostics and must not be used as the headline paper score.

To rebuild grouped summaries from an existing result file:

```bash
python scripts/aggregate_hf_results.py \
  --input outputs/hf_smoke.json \
  --output outputs/hf_smoke_grouped.json
```
