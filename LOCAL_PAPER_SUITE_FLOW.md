# Local Paper Suite Flow

This document explains what happens when you run:

```bash
python scripts/run_local_paper_suite.py \
  --profile main \
  --max-samples 50 \
  --sample-shards 5 \
  --resume
```

`run_local_paper_suite.py` is an orchestration wrapper. It does not implement
the KV-cache compression itself. Its job is to create a structured experiment
plan, call the existing HuggingFace runner, checkpoint every job, and then build
paper-friendly result tables.

## High-Level Flow

```text
CLI command
  -> build experiment jobs
  -> split jobs by model, dataset, and shard
  -> call scripts/run_hf_grid.py for each job
  -> run model prefill, TDC-KV compression, generation, and evaluation
  -> write raw JSON result files and checkpoints
  -> summarize JSON files into CSV/JSON/Markdown artifacts
```

## What Goes In

### Models

By default, the local suite runs two small instruction models:

```text
Qwen/Qwen2.5-1.5B-Instruct
HuggingFaceTB/SmolLM2-1.7B-Instruct
```

You can override them with:

```bash
--models "model_a,model_b"
```

### Datasets

The `main` profile uses:

```text
gsm8k
hotpotqa
niah_3072_d10
niah_3072_d50
niah_3072_d90
```

Meaning:

```text
GSM8K       -> grade-school math reasoning
HotPotQA    -> multi-hop question answering over context
NIAH d10    -> needle placed near 10% depth
NIAH d50    -> needle placed near 50% depth
NIAH d90    -> needle placed near 90% depth
```

For NIAH, the suite generates synthetic records locally. The default context is
3072 tokens so it is still long enough to stress KV cache behavior but practical
on a laptop GPU.

### Compression Ratios

The default `main` profile asks for these KV retention ratios:

```text
0.75, 0.5, 0.25, 0.125
```

They correspond to:

```text
0.75  -> keep 75% KV, compress 25%
0.5   -> keep 50% KV, compress 50%
0.25  -> keep 25% KV, compress 75%
0.125 -> keep 12.5% KV, compress 87.5%
```

### TDC-KV Algorithm Parameters

Default values:

```text
theta = 0.3
alpha = 0.6
beta = 0.4
recent_window = 16
dependency_top_k = 8
max_chunk_tokens = 64
min_chunk_tokens = 5
```

What they mean:

```text
theta
  Fraction of eligible chunks that become Tier 1 soft-protected chunks.
  Example: theta=0.3 protects the top 30% non-hard-protected chunks.

alpha
  Weight for direct attention-mass score.

beta
  Weight for dependency-routing score. beta = 1 - alpha.

recent_window
  Number of latest tokens used as the observation window for scoring.
  These recent tokens are also hard-protected by default.

dependency_top_k
  Number of dependency edges kept per chunk in the sparse dependency graph.

max_chunk_tokens
  Maximum size of a semantic chunk before the chunker forces a boundary.

min_chunk_tokens
  Minimum preferred chunk size before punctuation boundaries can close a chunk.
```

The `param_sweep` profile varies:

```text
theta: 0.2, 0.3, 0.4
alpha: 0.25, 0.6, 0.75
recent_window: 16, 32
retention: 0.5, 0.25, 0.125
```

Use that profile when you want ablation-style tables for the paper.

## What Jobs Are Created

With:

```bash
--profile main --sample-shards 5
```

the suite creates separate jobs for:

```text
each model
each dataset
each shard index from 0 to 4
```

Example output job names:

```text
main_qwen_qwen2_5_1_5b_instruct_gsm8k_shard_1_of_5
main_qwen_qwen2_5_1_5b_instruct_niah_3072_d50_shard_3_of_5
main_huggingfacetb_smollm2_1_7b_instruct_hotpotqa_shard_5_of_5
```

Each job calls:

```text
scripts/run_hf_grid.py
```

with a single model, a single dataset, and one shard. This makes the experiment
resumable and prevents one failed dataset/model pair from mixing with others.

## What Happens Inside Each Job

Inside `run_hf_grid.py`, the runner does the actual experiment:

1. Loads the model and tokenizer.
2. Loads or generates dataset records.
3. Builds the prompt for GSM8K, HotPotQA, or NIAH.
4. Tokenizes the prompt.
5. Runs a HuggingFace prefill pass.
6. Collects attention over bounded prefill blocks.
7. Builds chunks from the prompt tokens.
8. Builds a sparse dependency graph from attention.
9. Computes TDC-KV chunk scores.
10. Assigns protection tiers.
11. Evicts KV-cache tokens until the target budget is reached.
12. Generates an answer using the compressed cache.
13. Scores the answer and cache behavior.
14. Appends one run row to the checkpoint/result JSON.

## TDC-KV Compression Step

### Chunk Scoring

TDC-KV computes two signals:

```text
attention score
  How much recent queries attend to each chunk.

dependency score
  Whether a historical chunk is connected to chunks that are currently useful.
```

Then it fuses them:

```text
chunk_score = alpha * attention_score + beta * dependency_score
```

### Protection Tiers

Chunks are assigned to tiers:

```text
Tier 2
  Hard-protected chunks. Usually sink token chunk and recent-window chunks.

Tier 1
  Soft-protected important chunks selected using theta.

Tier 0
  Normal eviction candidates.
```

### Eviction

The evictor removes low-score chunks in this order:

```text
Tier 0 first
Tier 1 next if needed
Tier 2 only if fallback is allowed and the budget still cannot be met
```

The result is a smaller KV cache with:

```text
kept_indices
removed_indices
new_k_cache
new_v_cache
```

## What Comes Out

### Raw Job Files

Written under:

```text
outputs/local_paper/
```

Examples:

```text
main_qwen_qwen2_5_1_5b_instruct_gsm8k_shard_1_of_5.json
main_qwen_qwen2_5_1_5b_instruct_gsm8k_shard_1_of_5.checkpoint.json
suite_manifest.json
```

The checkpoint file is updated during the run. If you stop execution and rerun
the same command with `--resume`, completed rows are reused.

### Raw JSON Structure

Each result JSON contains:

```text
models
datasets
grid
environment
summary
grouped_results
runs
```

Important parts:

```text
grid
  The exact parameters used: methods, budget ratios, theta, alpha, windows, etc.

runs
  One row per model/dataset/sample/method/parameter/budget result.

grouped_results
  Aggregated summaries grouped by model, dataset, method, budget, and config.

summary
  Total successful/failed runs and high-level QA/cache summaries.
```

## Final Summary Artifacts

After jobs finish, the local suite calls:

```text
scripts/summarize_local_paper_results.py
```

This reads all successful raw JSON runs and writes:

```text
outputs/local_paper/artifacts/algorithm_parameter_grid.csv
outputs/local_paper/artifacts/algorithm_parameter_grid.json
outputs/local_paper/artifacts/algorithm_parameter_grid.md
outputs/local_paper/artifacts/artifact_manifest.json
```

The CSV is the main table you should inspect for paper results.

## How Output Is Evaluated

### Task Quality Metrics

The metric depends on the dataset:

```text
GSM8K
  metric = gsm8k_accuracy
  Meaning: extracted final numeric answer matches the gold answer.

HotPotQA
  metric = hotpotqa_f1
  Meaning: normalized answer-token F1 against the gold answer.

NIAH
  metric = niah_retrieval_accuracy
  Meaning: generated text contains the secret needle exactly after normalization.
```

Example:

```text
quality_mean = 0.64
```

means the average task score for that model/dataset/method/configuration was
0.64.

### Cache Compression Metrics

Important columns:

```text
requested_retention_ratio
  The target fraction of KV tokens to keep.

requested_compression_ratio
  1 - requested_retention_ratio.

actual_retention_ratio
  What fraction was actually kept after eviction.

actual_compression_ratio
  What fraction was actually removed.

budget_utilization
  kept_tokens / target_budget. Should be close to 1.0.

budget_shortfall_max
  Largest number of tokens by which the run underfilled the target budget.

budget_overflow_max
  Largest number of tokens by which the run exceeded the target budget.
```

Good example:

```text
requested_retention_ratio = 0.25
requested_compression_ratio = 0.75
budget_utilization = 0.998
budget_overflow_max = 0
```

This means the run compressed about 75% of the KV cache and matched the budget.

Bad example:

```text
budget_utilization = 0.55
budget_overflow_max = 128
```

This means the compression budget was not enforced well for that row.

### Runtime Metrics

Important columns:

```text
prefill_ms
  Time for the shared model prefill.

scoring_ms
  Time spent computing method-specific scores.

policy_ms
  Time spent selecting which KV tokens to keep/remove.

decode_ms
  Time spent generating output tokens after compression.

decode_tokens_per_second
  Generation throughput.

decode_ms_per_token
  Average decode latency per generated token.
```

### Memory Metrics

Important columns:

```text
kv_gib_before
  KV-cache memory before compression.

kv_gib_after
  KV-cache memory after compression.

kv_gib_saved
  KV-cache memory saved by compression.
```

Example:

```text
kv_gib_before = 0.80
kv_gib_after = 0.20
kv_gib_saved = 0.60
```

This means the compressed cache saved about 0.60 GiB for that group.

### Structural Metrics

Important columns:

```text
evidence_token_retention
  Fraction of known answer/evidence tokens that survived eviction.

evidence_chunk_survival
  Fraction of chunks containing known evidence that survived eviction.

tier0_chunks
  Average number of chunks in the normal eviction tier.

tier1_chunks
  Average number of soft-protected chunks.

tier2_chunks
  Average number of hard-protected chunks.
```

High evidence retention is good, especially for HotPotQA and NIAH.

## What Indicates a Successful Run

In the raw JSON:

```text
summary.failed_runs = 0
summary.successful_runs > 0
```

In the artifact CSV:

```text
quality_mean is not empty
budget_utilization is close to 1.0
budget_overflow_max is 0
actual_compression_ratio roughly matches requested_compression_ratio
kv_gib_saved is positive for compressed methods
```

Useful row:

```text
method = tdc_kv
requested_retention_ratio = 0.25
quality_mean = 0.58
budget_utilization = 1.0
budget_overflow_max = 0
kv_gib_saved > 0
```

Suspicious row:

```text
quality_mean = NA
budget_utilization = 0.4
budget_overflow_max > 0
kv_gib_saved = 0
```

That row should not be treated as a clean paper result.

## Common Profiles

### Smoke

```bash
python scripts/run_local_paper_suite.py --profile smoke --resume
```

Purpose:

```text
Smallest check that model loading, dataset loading, compression, generation,
checkpointing, and summarization work.
```

### Main

```bash
python scripts/run_local_paper_suite.py \
  --profile main \
  --max-samples 50 \
  --sample-shards 5 \
  --resume
```

Purpose:

```text
Primary quality-vs-compression table for the paper.
```

### Parameter Sweep

```bash
python scripts/run_local_paper_suite.py \
  --profile param_sweep \
  --max-samples 20 \
  --sample-shards 5 \
  --resume
```

Purpose:

```text
Shows how theta, alpha, recent window, and compression ratio affect quality and
KV-cache savings.
```

### Baselines

```bash
python scripts/run_local_paper_suite.py \
  --profile baselines \
  --max-samples 50 \
  --sample-shards 5 \
  --resume
```

Purpose:

```text
Compares TDC-KV against fullkv, StreamingLLM, H2O, SnapKV, and ChunkKV.
```

## Resume Behavior

Every job has:

```text
<job>.checkpoint.json
<job>.json
```

If the run is stopped midway, rerun the exact same command with `--resume`.

Completed rows in the checkpoint are skipped. If a final output JSON already
exists and has zero failed runs, the local suite skips that completed job.

Do not change these when resuming:

```text
models
datasets
max_samples
sample_shards
budget_ratios
theta/alpha/recent_window grids
max_length
max_new_tokens
```

Changing them changes the experiment fingerprint and the checkpoint will not
match.
