# TDC-KV Experiment Runbook

Last updated: 2026-08-17

This is the command-level runbook for generating TDC-KV development,
qualification, tuning, headline, timing, and ablation results. It also states
the exact experiment specification behind every command.

The two intended execution environments are:

- **Laptop:** NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM. Use it for tests,
  the 0.5B qualification suite, and optional 1.5B/small-model experiments.
- **Paper hardware:** A100-class CUDA runtime, preferably an A100 40 GB or
  larger. Use it for the FP16 7B/8B tuning, headline, timing, and ablation jobs.

The paper correctness path uses eager attention and FP16 weights. Quantized
3B/7B/8B execution is not implemented and must not be mixed with the FP16
headline results.

## 1. Experiment families

| Experiment | Command profile | Hardware | Purpose | Paper role |
|---|---|---|---|---|
| Unit verification | `python -m pytest` | RTX 4050 or CPU | Verify algorithms, cache adapters, reporting, and launchers | Required before GPU runs |
| Quick pipeline check | `qualification --max-samples 1` | RTX 4050 | One end-to-end sample from each qualification dataset | Smoke/qualification only |
| Full qualification | `qualification` | RTX 4050 | Five samples per qualification dataset with every method | Required gate, not headline evidence |
| Local small-model scaling | `main --models ... --max-samples ...` | RTX 4050 | Optional 1.5B or feasible 2B experiment | Supplementary/development |
| Hyperparameter tuning | `tuning` | A100-class | Select one frozen TDC-KV configuration | Configuration selection only |
| Headline quality | `main` | A100-class | Final partition, three model families, all methods and budgets | Main paper tables and significance tests |
| Timing repetitions | `timing` | A100-class | Three repeated measurements on controlled 8K NIAH | Efficiency table |
| Ablations | `ablations` | A100-class | Test signals, tiers, chunking, and layer weighting | Ablation tables/figures |
| Complete suite | `all` | A100-class | Qualification → tuning → main → timing → ablations | Recommended paper workflow |

## 2. One-time environment preparation

Run these commands from the repository root.

### 2.1 Install the constrained dependencies

Keep the CUDA-enabled PyTorch build supplied by the laptop or Colab runtime.
Then install the repository packages:

```powershell
pip install -r requirements.txt -c constraints-paper.txt
pip install -e .
```

The constrained paper environment uses Python 3.12, eager attention, FP16,
deterministic algorithms, and a 90% VRAM preflight safety limit. The precise
PyTorch, CUDA, cuDNN, NVIDIA driver, GPU, package, and Git versions are stored
inside every result JSON.

### 2.2 Verify CUDA and the repository

```powershell
nvidia-smi
python -m pytest -q -p no:cacheprovider
```

Expected repository result at the time this runbook was written:

```text
139 passed
```

Do not start a paper run if CUDA is missing, tests fail, or `nvidia-smi` does
not show the intended GPU.

## 3. Hugging Face authentication and immutable model revisions

Meta Llama is gated. The Hugging Face account associated with the token must
have accepted the model license.

PowerShell:

```powershell
$env:HF_TOKEN="hf_your_read_token"
```

Bash/Colab:

```bash
export HF_TOKEN="hf_your_read_token"
```

Never write the token into a command file, JSON manifest, notebook committed to
Git, or result artifact.

Resolve the exact immutable model commits. The optional 1.5B model is included
so the same revision file can also drive laptop experiments:

```powershell
python scripts/preflight_hf_models.py --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2.5-7B-Instruct" --output protocol/model_revisions.json
```

This command checks repository access and writes model ID → commit SHA mappings
to `protocol/model_revisions.json`. It does not download all model weights.

## 4. Freeze disjoint datasets

Run this exactly once for a paper protocol:

```powershell
python scripts/run_paper_suite.py --profile qualification --freeze-manifest
```

It creates `protocol/paper_dataset_manifest.json` and stops. The frozen
selection policy is:

| Partition | Records per dataset | Used by |
|---|---:|---|
| `qualification` | 5 | Small-model qualification |
| `tuning` | 50 | Tuning, timing, and ablations |
| `final` | 200 | Headline quality experiment |

The partitions are disjoint sequential selections with record identities and
SHA-256 hashes. If an upstream dataset changes, the experiment fails rather
than silently selecting different records.

Review and commit both protocol files:

```powershell
git add protocol/model_revisions.json protocol/paper_dataset_manifest.json
git commit -m "Freeze paper model and dataset revisions"
git status --short
```

`git status --short` must produce no output before a strict paper run. The
qualification gate deliberately rejects a dirty worktree.

## 5. Inspect the complete launch plan without running models

```powershell
python scripts/run_paper_suite.py --profile all --model-revisions-file protocol/model_revisions.json --sample-shards 10 --output-root outputs/paper_dry_run --dry-run
```

This writes a suite manifest and prints every child command, but it does not
load datasets or model weights. Inspect:

```text
outputs/paper_dry_run/suite_manifest.json
```

## 6. RTX 4050 experiments

### 6.1 Quick one-sample pipeline check

Run this first on the laptop:

```powershell
python scripts/run_paper_suite.py --profile qualification --methods tdc_kv --max-samples 1 --model-revisions-file protocol/model_revisions.json --output-root outputs/qualification_quick_tdc --resume
```

Specification:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Datasets | GSM8K, HotpotQA distractor, NIAH 1K at 50% depth |
| Samples | 1 per dataset |
| Methods | TDC-KV only |
| KV retention | 50%, 25% |
| Prompt limit | 2,048 tokens; truncation forbidden |
| Generation | 64 new tokens, greedy |
| Attention | Last layer, eager implementation |
| Prefill block | 16 queries |
| TDC-KV defaults | `alpha=0.6`, `theta=0.3`, recent window 16 |
| Decode policy | Common streaming policy for compressed methods |
| Parity | Native FullKV vs controlled unpruned cache on first sample per dataset |

Expected successful rows: `3 datasets × 1 method × 2 budgets = 6`.
The native-vs-controlled FullKV parity check still runs once per dataset as a
correctness control, but FullKV is not included as an experiment method row.

The result is accepted only if CUDA, model preflight, immutable revisions,
frozen data hashes, cache parity, exact budgets, runtime/memory fields, exact
NIAH length, evidence localization, and no prompt truncation all pass.

### 6.2 Full laptop qualification

After the one-sample check succeeds:

```powershell
python scripts/run_paper_suite.py --profile qualification --methods tdc_kv --model-revisions-file protocol/model_revisions.json --output-root outputs/qualification_full_tdc --resume
```

This uses five samples per dataset and produces 30 TDC-KV rows. It is a much
stronger pipeline qualification, but it is not a substitute for the 7B/8B
headline experiment.

### 6.3 Optional local 1.5B scaling run

The safest larger local model in the currently covered Qwen family is the 1.5B
variant. Start with one sample per final dataset:

```powershell
python scripts/run_paper_suite.py --profile main --methods tdc_kv --models Qwen/Qwen2.5-1.5B-Instruct --max-samples 1 --sample-shards 1 --model-revisions-file protocol/model_revisions.json --output-root outputs/local_qwen_1p5b --resume
```

This uses the main protocol: GSM8K, HotpotQA, three tokenizer-exact NIAH 8K
depths, four retention budgets, TDC-KV only, FP16, 9,216-token prompt limit,
128-token generation, and strict qualification. It uses the default TDC-KV
configuration unless `--selected-config outputs/paper/selected_config.json` is
provided.

Increase `--max-samples` gradually to 5, 10, or 20 only after the one-sample
run passes the VRAM preflight and completes without CUDA out-of-memory errors.

A nominal 2B FP16 model has approximately 4 GB of weights before KV cache,
attention blocks, activations, and CUDA overhead. It may or may not fit at 8K.
The built-in preflight is the authority. A 3B FP16 model requires approximately
6 GB for weights alone and is not a reliable target for a 6 GB RTX 4050.

Do not report a quantized 3B run as directly comparable with FP16 headline
results. Quantized loading is not implemented in the current paper path.

## 7. Individual A100-class experiment profiles

Use these commands when running the paper phases separately. The `all` profile
in Section 8 is preferred because it automatically applies the tuned
configuration to later phases.

### 7.1 Tuning

```bash
python scripts/run_paper_suite.py --profile tuning --model-revisions-file protocol/model_revisions.json --output-root outputs/paper --resume
```

Specification:

| Setting | Value |
|---|---|
| Model | First headline model: Meta-Llama-3-8B-Instruct |
| Data | Tuning partition of GSM8K, HotpotQA, NIAH 8K at 10/50/90% depth |
| Samples | 20 per dataset; 100 total |
| Methods | FullKV and TDC-KV |
| Retention | 25%, 12.5% |
| `theta` | 0.2, 0.3, 0.4 |
| Recent window | 16, 32 |
| `alpha` | 0.25, 0.5, 0.6, 0.75, 1.0 |
| TDC-KV points | 60 per sample |
| Rows | 6,100 total: 60 TDC-KV + 1 FullKV per sample |
| Selection | Highest macro-dataset task score; policy time breaks ties |

If profiles are run separately, freeze the selected configuration explicitly:

```bash
python scripts/select_tuning_config.py --input outputs/paper/tuning.json --output outputs/paper/selected_config.json
```

### 7.2 Headline main experiment

```bash
python scripts/run_paper_suite.py --profile main --methods tdc_kv --model-revisions-file protocol/model_revisions.json --selected-config outputs/paper/selected_config.json --sample-shards 10 --output-root outputs/paper --resume
```

Specification:

| Setting | Value |
|---|---|
| Models | Llama-3 8B Instruct, Mistral 7B Instruct v0.3, Qwen2.5 7B Instruct |
| Data | Final partition: GSM8K, HotpotQA, NIAH 8K at 10/50/90% depth |
| Samples | 200 per dataset, 1,000 per model, 3,000 model-samples total |
| Methods | TDC-KV only; baseline results are supplied separately |
| Retention | 50%, 25%, 12.5%, 6.25% |
| Generation | Greedy, 128 new tokens |
| Prompt maximum | 9,216 tokens; truncation forbidden |
| Precision/backend | FP16, eager attention |
| Seed | 42; deterministic algorithms enabled |
| Rows per sample | 4: TDC-KV at four budgets |
| Total rows | 12,000 successful TDC-KV rows expected |

The main experiment is split by model, dataset, and sample shard. Ten shards
means 20 final samples per child job. Sharding changes job boundaries only; it
does not reduce the 200-sample final dataset or alter statistics.

### 7.3 Timing repetitions

```bash
python scripts/run_paper_suite.py --profile timing --methods tdc_kv --model-revisions-file protocol/model_revisions.json --selected-config outputs/paper/selected_config.json --sample-shards 10 --output-root outputs/paper --resume
```

Specification:

| Setting | Value |
|---|---|
| Models | All three headline models |
| Data | NIAH 8K, 50% depth, tuning partition |
| Samples | 20 per model per repetition |
| Repetitions/seeds | 3: 13, 42, 101 |
| Methods | TDC-KV only |
| Retention | 25% |
| Rows | 180 TDC-KV rows total |
| Reported efficiency | Shared prefill, method-specific scoring/policy/decode, decode ms/token, tokens/s, peak VRAM, KV bytes before/after |

Timing rows are deliberately separate from final-partition quality rows.

### 7.4 Ablations

```bash
python scripts/run_paper_suite.py --profile ablations --model-revisions-file protocol/model_revisions.json --selected-config outputs/paper/selected_config.json --sample-shards 10 --output-root outputs/paper --resume
```

Specification:

| Setting | Value |
|---|---|
| Model | First headline model: Meta-Llama-3-8B-Instruct |
| Data | HotpotQA and NIAH 8K at 50% depth, tuning partition |
| Samples | 50 per dataset |
| Methods | FullKV and TDC-KV |
| Retention | 25%, 12.5% |
| Generation | 64 new tokens |

The ten ablation variants are:

| Variant | Change |
|---|---|
| `signal` | Sweep `alpha`: 0, 0.25, 0.5, 0.6, 0.75, 1.0 |
| `no_tier1` | Disable dependency-ranked Tier 1 |
| `no_sink` | Disable sink-chunk protection |
| `no_recent` | Disable recent-chunk protection |
| `fixed_8` | Fixed-width 8-token chunks |
| `fixed_16` | Fixed-width 16-token chunks |
| `fixed_32` | Fixed-width 32-token chunks |
| `token_level` | One token per chunk |
| `layers_uniform` | All layers with exactly uniform weights |
| `layers_linear` | All layers with linearly increasing weights |

The all-layer variants are scoring ablations. They are not PyramidKV-style
per-layer budget allocation.

## 8. Optimal command for all paper-grade results

After dependencies, tests, Hugging Face authentication, frozen protocol files,
a clean Git worktree, and the RTX 4050 qualification have all succeeded, move
the repository to an A100-class runtime and run:

```bash
python scripts/run_paper_suite.py --profile all --methods tdc_kv --model-revisions-file protocol/model_revisions.json --sample-shards 10 --output-root outputs/paper --resume
```

This is the recommended complete command. Do not add `--max-samples`; doing so
reduces the frozen paper sample counts and creates a pilot rather than the final
paper experiment.

Why this is the preferred command:

1. It runs the 0.5B qualification gate first.
2. It performs tuning only on the disjoint tuning partition.
3. It writes `outputs/paper/selected_config.json`.
4. It automatically applies the selected configuration to pending main,
   timing, and ablation jobs.
5. It runs the 200-sample final partition exactly once for headline quality.
6. It performs three separate timing repetitions.
7. It produces isolated, resumable model/dataset/sample-shard checkpoints.
8. It skips already completed qualified outputs when the same command is
   restarted with `--resume`.

Ten sample shards is the recommended reliability setting for interruptible
Colab sessions. On a stable dedicated A100 server, five shards reduces model
reload overhead while preserving the same samples:

```bash
python scripts/run_paper_suite.py --profile all --methods tdc_kv --model-revisions-file protocol/model_revisions.json --sample-shards 5 --output-root outputs/paper --resume
```

For Colab, place `--output-root` on persistent storage if runtime loss would
otherwise delete results. Keep the same absolute output location on every
resume.

## 9. Output layout and recovery

The suite writes:

```text
outputs/paper/
├── qualification.json
├── tuning.json
├── selected_config.json
├── main_<model>_<dataset>_shard_*.json
├── timing_rep_<n>_<model>_<dataset>_shard_*.json
├── ablation_<variant>_<model>_<dataset>_shard_*.json
├── *.checkpoint.json
└── suite_manifest.json
```

After interruption, rerun the identical command with `--resume`. Do not change
models, revisions, package environment, Git commit, dataset manifest, seed,
budgets, prompt settings, shard count, or output root while resuming. The
experiment fingerprint rejects incompatible checkpoints.

If the code, environment, model revision, dataset manifest, or experiment grid
must change, use a new output root. Do not merge incompatible result JSONs.

## 10. Generate paper artifacts

Generate each experimental family separately. This prevents final-partition
quality results from being mixed with timing/tuning-partition repetitions.

### 10.1 Headline tables, curves, and paired significance tests

```bash
python scripts/generate_paper_artifacts.py --inputs "outputs/paper/main_*.json" --output-dir outputs/paper/artifacts/main
```

### 10.2 Repeated timing and memory tables

```bash
python scripts/generate_paper_artifacts.py --inputs "outputs/paper/timing_rep_*.json" --output-dir outputs/paper/artifacts/timing
```

### 10.3 Ablation tables and figures

```bash
python scripts/generate_paper_artifacts.py --inputs "outputs/paper/ablation_*.json" --output-dir outputs/paper/artifacts/ablations
```

The artifact generator refuses failed or unqualified input files and records
the SHA-256 hash and experiment fingerprint of every input JSON.

Headline metrics are:

- GSM8K extracted numeric accuracy under the frozen eight-shot protocol.
- NIAH exact normalized needle retrieval accuracy.
- HotpotQA official-style normalized answer F1.
- Matched KV retention and compression.
- Sample-level mean/std/bootstrap confidence intervals.
- Paired TDC-KV versus baseline tests with Holm-adjusted p-values.
- Decode milliseconds/token, tokens/second, method-specific overhead, peak
  VRAM, and physical KV bytes before/after eviction.
- Evidence-token retention/eviction, evidence-chunk survival, observed evidence
  depth, and head consensus.

The evidence-token eviction value is a global shared-mask proxy. It must not be
described as true head/layer reachability GER. Likewise, the bundled
StreamingLLM, H2O, SnapKV, and ChunkKV policies are matched-budget local
approximations, not bit-for-bit vendored official implementations.

## 11. Paper-grade acceptance checklist

A result set is paper-ready only when all of the following are true:

- The repository test suite passes.
- The intended NVIDIA GPU and driver are recorded.
- Git commit provenance exists and the worktree was clean.
- Model IDs were requested at immutable commit SHAs.
- Every dataset used a hash-verified frozen manifest partition.
- Qualification, tuning, and final data did not overlap.
- Every requested model × dataset × sample × method × budget/configuration
  point completed.
- Native FullKV and controlled unpruned-cache parity passed on the configured
  controls.
- No prompt was truncated.
- NIAH context lengths and needle positions were tokenizer-exact.
- Every compressed row had zero budget overflow, at least 99% utilization, and
  at most one token of shortfall.
- Runtime stages and KV-memory fields were finite and present.
- Headline, timing, and ablation artifacts were generated separately.
- No 0.5B/1.5B laptop result was substituted for the 7B/8B headline result.
