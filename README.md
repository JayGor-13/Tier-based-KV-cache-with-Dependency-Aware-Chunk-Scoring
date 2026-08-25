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
pip install -r requirements.txt -c constraints-paper.txt
pip install -e .
```

On Colab, keep the runtime-provided PyTorch build. The text-only pipeline does
not require `torchvision` or `torchaudio`; remove those optional wheels if their
CUDA build differs from PyTorch. The bundled quickstart performs this cleanup
before importing Transformers.

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

## Local 1-2B Paper Sweep

Use this path for a laptop RTX 4000-series GPU. It reuses
`scripts/run_hf_grid.py`, but standardizes a resumable local research layout for
two 1-2B instruction models, GSM8K, HotPotQA, and NIAH:

- default models: `Qwen/Qwen2.5-1.5B-Instruct` and
  `HuggingFaceTB/SmolLM2-1.7B-Instruct`;
- default datasets: GSM8K test, HotPotQA distractor validation, and NIAH at
  10%, 50%, and 90% needle depths with a 3072-token local context;
- default compression sweep: 50%, 30%, 20%, and 10% KV retention;
- default algorithm parameters: `theta=0.3`, `recent_window=16`, `alpha=0.6`,
  `dependency_top_k=8`, semantic chunks up to 64 tokens;
- outputs: raw resumable JSON files plus CSV/JSON/Markdown summaries grouped by
  model, dataset, method, algorithm parameters, and compression ratio.

First run the tiny end-to-end smoke matrix:

```bash
python scripts/run_local_paper_suite.py --profile smoke --resume
```

Then run a main quality/compression sweep:

```bash
python scripts/run_local_paper_suite.py \
  --profile main \
  --max-samples 50 \
  --sample-shards 5 \
  --resume
```

Run the TDC-KV parameter sweep separately so it can be reported as an ablation
table without mixing it with the default setting:

```bash
python scripts/run_local_paper_suite.py \
  --profile param_sweep \
  --max-samples 20 \
  --sample-shards 5 \
  --resume
```

The suite writes job files under `outputs/local_paper/` and summary artifacts
under `outputs/local_paper/artifacts/`:

- `algorithm_parameter_grid.csv`: spreadsheet-friendly paper table;
- `algorithm_parameter_grid.json`: machine-readable aggregate table;
- `algorithm_parameter_grid.md`: quick Markdown table for inspection;
- `artifact_manifest.json`: input paths and SHA-256 hashes.

The suite automatically passes its explicit completed output paths to the
summarizer. For a manual diagnostic rebuild, list files explicitly; wildcard
discovery requires the deliberately non-paper `--allow-glob` override:

```bash
python scripts/summarize_local_paper_results.py \
  --inputs "outputs/local_paper/*.json" \
  --allow-glob \
  --output-dir outputs/local_paper/artifacts
```

For stricter paper provenance, resolve model revisions and optionally freeze a
local dataset manifest before the final run:

```bash
python scripts/preflight_hf_models.py \
  --models "Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct" \
  --output protocol/local_model_revisions.json

python scripts/run_local_paper_suite.py --freeze-manifest

python scripts/run_local_paper_suite.py \
  --profile main \
  --model-revisions-file protocol/local_model_revisions.json \
  --use-frozen-manifest \
  --require-qualified \
  --require-cuda \
  --require-model-preflight \
  --require-model-revision \
  --require-no-truncation \
  --resume
```

## Paper Qualification Workflow

The paper path now enforces the following contracts:

- prompts are serialized exactly once (`raw`, `chat`, or `auto`) and their raw,
  serialized, and token-id SHA-256 hashes are stored;
- the optional FullKV parity check compares HuggingFace generation with the
  unpruned custom-cache path on the exact same input token ids;
- the controlled FullKV and compressed methods share the same measured prefill;
  method-specific scoring, policy, and decode time are reported separately with
  CUDA synchronization, per-token decode throughput, peak VRAM, and physical KV
  tensor bytes before/after eviction;
- H2O and SnapKV scores are computed once and passed into eviction, while
  ChunkKV uses direct attention scores independently of TDC-KV dependency
  routing;
- `common_streaming` applies one decode-cache policy to all compressed methods;
- every successful row has hierarchical deterministic identities, and a
  transactional SQLite checkpoint is updated after every completed row;
- non-finite logits, attentions, scores, K/V tensors, runtime values, and final
  JSON scalars are rejected rather than repaired or serialized as `NaN`;
- frozen record-level dataset manifests keep qualification, tuning, and final
  partitions disjoint and hash-verified;
- tokenizer-exact NIAH construction records actual context length/depth, while
  the qualification gate rejects prompt truncation;
- model access, immutable Hugging Face revisions, full-context support, eager
  attention compatibility, CUDA, and estimated VRAM fit are preflighted;
- the qualification gate also rejects failed rows, parity failures, empty
  generations, missing measurements, incomplete method/budget coverage, dirty
  Git state, and matched-budget violations.

Use the local RTX 4050 (6 GB) for tests and a small-model qualification run:

```bash
python scripts/run_hf_grid.py \
  --models Qwen/Qwen2.5-0.5B-Instruct \
  --datasets "name=gsm8k,source=openai/gsm8k,config=main,split=test,adapter=gsm8k,protocol=chunkkv_gsm8k_8shot,prompt_field=question,answer_field=answer" \
  --methods fullkv,streamingllm,h2o,snapkv,chunkkv,tdc_kv \
  --budget-ratios 0.5 \
  --max-samples 2 \
  --max-length 1024 \
  --max-new-tokens 64 \
  --prefill-block-size 32 \
  --dtype bfloat16 \
  --decode-policy common_streaming \
  --prompt-serialization raw \
  --checkpoint outputs/phase1_local.checkpoint.sqlite \
  --output outputs/phase1_local.json
```

Resume the exact grid after interruption by adding `--resume` with the same
arguments. A changed model, dataset, seed, protocol, or grid is rejected rather
than mixed into an existing checkpoint.

The paper loader uses unquantized BF16 weights and eager attention because
attention tensors are required for scoring. A 7B/8B model therefore does not
fit the 6 GB laptop GPU. Run those paper experiments on a Colab Pro session only
after confirming an A100-class runtime (preferably 40 GB or more); Colab Pro
does not guarantee a particular GPU. Keep `--prefill-block-size 32` for the
first 7B/8B smoke and increase it only after inspecting recorded peak VRAM.

The local StreamingLLM, H2O, SnapKV, and ChunkKV implementations are explicitly
tagged as `approximation` in `grid.method_metadata`. Do not describe them as
bit-for-bit official reference implementations in the paper.

## Complete Paper Experiment Pipeline

For the full command-by-command workflow and the exact specification of every
experiment profile, see [`EXPERIMENT_RUNBOOK.md`](EXPERIMENT_RUNBOOK.md).

First authenticate with a Hugging Face read token. Meta Llama also requires the
account to have accepted the model license. In PowerShell:

```powershell
$env:HF_TOKEN="hf_your_read_token"
```

Resolve and save immutable commits before downloading model weights:

```bash
python scripts/preflight_hf_models.py --models "Qwen/Qwen2.5-0.5B-Instruct,meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2-7B-Instruct" --output protocol/model_revisions.json
```

Freeze the dataset protocol once. This command only writes the manifest and
then stops:

```bash
python scripts/run_paper_suite.py --profile qualification --freeze-manifest
```

Review and commit `protocol/model_revisions.json` and
`protocol/paper_dataset_manifest.json`; final jobs require a clean worktree. Then
run a one-sample, all-method laptop qualification on the RTX 4050:

```bash
python scripts/run_paper_suite.py --profile qualification --models Qwen/Qwen2.5-0.5B-Instruct --methods fullkv,tdc_kv --max-samples 1 --model-revisions-file protocol/model_revisions.json
```

The 7B/8B eager-BF16 correctness path does not fit a 6 GB RTX 4050. Use an
A100-class Colab Pro runtime for tuning/final experiments, verify the GPU shown
by `nvidia-smi`, and run the resumable sharded suite:

```bash
python scripts/run_paper_suite.py --profile all --methods tdc_kv --model-revisions-file protocol/model_revisions.json --sample-shards 10 --resume
```

The `all` profile runs qualification, tuning, a 200-example multi-dataset pilot,
three independent timing repetitions, and ablations. It does not run the
official full GSM8K test. Tuning writes
`selected_config.json` and applies it to all subsequent jobs. Expensive jobs are
split by model, dataset, and sample shard, with isolated transactional SQLite
checkpoints.
Use `--dry-run` to inspect generated commands and `--resume` after interruption.

For the official 1,319-example GSM8K comparison, freeze its separate manifest
and run the dedicated profile after the pilot passes:

```bash
python scripts/run_paper_suite.py --profile gsm8k_full --protocol-manifest protocol/gsm8k_full_manifest.json --freeze-manifest
python scripts/run_paper_suite.py --profile gsm8k_full --protocol-manifest protocol/gsm8k_full_manifest.json --model-revisions-file protocol/model_revisions.json --selected-config outputs/paper/selected_config.json --sample-shards 20 --output-root outputs/gsm8k_full --resume
```

Once the result jobs finish, produce headline CSV/Markdown tables,
significance tests, and figures with:

```bash
python scripts/generate_paper_artifacts.py \
  --suite-manifest outputs/paper/suite_manifest.json \
  --job-prefix main \
  --output-dir outputs/paper/artifacts/main
```

Generate timing and ablation artifacts separately as documented in the
runbook; this avoids mixing final-partition quality with tuning-partition rows.

The artifact builder refuses unqualified result files and records input hashes.
Outputs include main results, quality curves, efficiency tables, paired
Wilcoxon/t-tests with Holm correction, sample-level bootstrap confidence
intervals, quality/compression curves, NIAH heatmaps, evidence-token eviction
curves, tier distributions, VRAM/throughput plots, and ablation charts.

Per-run structural logging now includes answer-critical token localization,
evidence-token retention, evidence-chunk survival, an explicitly labeled global
shared-mask evidence-token eviction proxy, observed evidence depth, and head
consensus/diversity. The current implementation does not claim a true
head/layer reachability GER or per-layer budget allocation. Ablation controls
expose semantic/fixed/token chunks, Tier-1 removal,
sink/recent protection removal, and uniform versus linearly weighted all-layer
attention.

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
