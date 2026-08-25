# TDC-KV Experiment Runbook

Last updated: 2026-08-25

This runbook separates numerical diagnosis, tuning, pilot evaluation, and the
official full GSM8K comparison. A run is not paper evidence merely because it
finished: it must satisfy the schema-v2, provenance, numerical-health, parity,
coverage, and qualification gates described below.

## 1. What counts as the comparison metric

For GSM8K, report all of the following at matched KV retention:

- absolute extracted final-answer accuracy on the official 1,319-example test
  split under the frozen eight-shot prompt;
- FullKV accuracy for the same model, prompt tokens, precision, and decoding
  contract;
- normalized quality ratio (NQR), defined as
  `100 × compressed accuracy / paired FullKV accuracy`;
- a 95% Wilson interval for accuracy;
- actual physical KV retention, not only the requested budget;
- paired sample-level differences and corrected significance tests.

Use 30%, 20%, and 10% KV retention for the headline curve. A 200-example
`main` profile is a multi-dataset pilot. It must not be presented as the
official full GSM8K score.

HotpotQA uses normalized answer F1, and NIAH uses exact normalized retrieval
accuracy. Evidence retention and global eviction are diagnostic structural
metrics, not replacements for task quality.

The bundled StreamingLLM, H2O, SnapKV, and ChunkKV implementations are marked
`approximation`. They support controlled matched-budget engineering comparisons,
but not a claim of reproducing an official paper implementation. A headline
claim against a published baseline requires its official repository, exact
commit, model, prompt, dataset, and conformance check.

## 2. Precision and the 8B-model decision

The paper profiles use unquantized BF16 weights and strict eager attention.
Using an 8B model that is “not FP16” is not sufficient by itself. The exact
model/device/backend combination must pass:

1. native greedy generation;
2. a whole-prompt eager forward;
3. unpruned custom-cache generation in one block;
4. unpruned custom-cache generation with block sizes 128 and 16;
5. finite logits, attentions, K/V tensors, and generated token IDs;
6. native/custom token parity and block-size invariance.

Unquantized BF16 7B/8B weights alone usually require roughly 14–16 GiB before
KV cache, eager-attention blocks, activations, and runtime overhead. They do not
fit a 6 GiB RTX 4050. Use that laptop for CPU tests and small-model diagnosis;
use a validated 24 GiB or preferably A100-class 40 GiB+ GPU for the headline
models. The built-in model preflight is authoritative.

## 3. Install and verify

```powershell
pip install -r requirements.txt -c constraints-paper.txt
pip install -e .
python -m pytest -q --basetemp tmp/pytest_paper
```

The test suite must pass before any GPU job. Result-producing paper jobs also
require a clean committed worktree because the qualification gate records and
checks the Git commit.

## 4. Quarantine old results

Audit legacy JSON/CSV files without modifying them:

```powershell
python scripts/validate_results.py "C:/path/to/result.json" "C:/path/to/summary.csv" --output outputs/result_audit.json --fail-on paper-ineligible
```

Legacy aggregate CSVs cannot become paper artifacts because they lack raw
predictions, sample identities, model revisions, protocol fingerprints, and
qualification evidence. Non-finite or degenerate successful rows are reported
as corrupt.

## 5. Run the A–H numerical diagnostic first

Start with three frozen GSM8K prompts on the 1.5B model:

```powershell
python scripts/diagnose_hf_numerics.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <commit-sha> --max-samples 3 --output outputs/diagnostics/qwen_1p5b_a_h.json
```

The repository includes `protocol/gsm8k_diagnostic_manifest.json`, containing
three hash-verified training examples specifically for this numerical check.

Then repeat the same command for every target 7B/8B model. Cases A–D use FP16;
E–H use BF16. FP16 failures are acceptable as a documented environment result.
Cases E–H must all pass before compression results are accepted. If E–H
disagree, fix the eager/blockwise/custom-cache path rather than running a larger
experiment.

## 6. Pin model revisions

```powershell
python scripts/preflight_hf_models.py --models "meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2-7B-Instruct" --output protocol/model_revisions.json
```

Review the resolved immutable commit SHAs. Do not use a moving branch or silently
substitute another model size.

## 7. Freeze datasets

Regenerate the multi-dataset pilot manifest after the train/test partition
changes in this implementation:

```powershell
python scripts/run_paper_suite.py --profile qualification --protocol-manifest protocol/paper_dataset_manifest.json --freeze-manifest
```

This freezes disjoint qualification, tuning, and 200-example final selections.
Qualification and tuning use training data; the pilot final selection uses test
or validation data.

Freeze the separate official full GSM8K manifest:

```powershell
python scripts/run_paper_suite.py --profile gsm8k_full --protocol-manifest protocol/gsm8k_full_manifest.json --freeze-manifest
```

This second manifest contains exactly all 1,319 official GSM8K test records in
its `final` partition. Review and commit both manifests and the model revisions
before strict execution.

## 8. Qualification and tuning

Inspect every generated child command without loading weights:

```powershell
python scripts/run_paper_suite.py --profile all --model-revisions-file protocol/model_revisions.json --sample-shards 10 --output-root outputs/paper_dry_run --dry-run
```

Run the target-model qualification on appropriate GPU hardware:

```powershell
python scripts/run_paper_suite.py --profile qualification --model-revisions-file protocol/model_revisions.json --output-root outputs/paper --resume
```

The qualification profile uses 10 training examples per dataset, every method,
greedy decoding, 50% and 25% retention, BF16 eager attention, all-layer scoring,
512 new tokens, exact FullKV/custom parity coverage, and no truncation.

Tune TDC-KV only on the frozen training partition:

```powershell
python scripts/run_paper_suite.py --profile tuning --model-revisions-file protocol/model_revisions.json --output-root outputs/paper --resume
```

Tuning uses FullKV and TDC-KV at 30% and 10% retention. It selects from the
declared theta, recent-window, and alpha grid and writes
`outputs/paper/selected_config.json`. Do not inspect official test answers while
choosing the configuration.

## 9. Pilot and full GSM8K execution

Run the 200-example multi-dataset pilot before the full evaluation:

```powershell
python scripts/run_paper_suite.py --profile main --model-revisions-file protocol/model_revisions.json --selected-config outputs/paper/selected_config.json --sample-shards 10 --output-root outputs/paper --resume
```

Inspect failures, output lengths, parseability, actual retention, NQR, and
FullKV pairing. If the protocol changes, use a new output root and rerun every
affected method; never merge incompatible checkpoints.

Run the frozen 1,319-example GSM8K comparison only after the pilot passes:

```powershell
python scripts/run_paper_suite.py --profile gsm8k_full --protocol-manifest protocol/gsm8k_full_manifest.json --model-revisions-file protocol/model_revisions.json --selected-config outputs/paper/selected_config.json --sample-shards 20 --output-root outputs/gsm8k_full --resume
```

The full profile runs FullKV, the four matched-budget local baseline
approximations, and TDC-KV at 30%, 20%, and 10% retention. Use `--methods
fullkv,tdc_kv` only for a clearly labeled algorithm-only study; it is not a
complete baseline table.

## 10. Resume and output integrity

Every new suite job uses a transactional SQLite checkpoint:

```text
<job>.checkpoint.sqlite
<job>.json
suite_manifest.json
```

Successful rows are immutable resume units keyed by the full execution,
sample, and method identities. Parity rows are keyed by model/dataset/sample.
The checkpoint fingerprint includes model revision, materialized records,
prompt/token hashes, precision, backend, seed, shard, and method configuration.
Changing any of those requires a new checkpoint. Existing schema-v2 JSON
checkpoints remain readable as a migration fallback.

All final JSON is RFC-compliant (`NaN` and `Infinity` are rejected) and written
atomically. A final artifact records state `complete`, exact coverage counts,
protocol/job fingerprints, per-row numerical and generation health, and the
recomputed qualification contract.

## 11. Generate artifacts from explicit manifests

Avoid recursive wildcard discovery. Select completed jobs from the suite
manifest:

```powershell
python scripts/generate_paper_artifacts.py --suite-manifest outputs/gsm8k_full/suite_manifest.json --job-prefix gsm8k_full --output-dir outputs/gsm8k_full/artifacts
```

For the pilot, timing, or ablation families, use their suite manifest and the
corresponding `main`, `timing_`, or `ablation_` job prefix. The loader validates
schema-v2, re-runs the declared qualification instead of trusting a stored
`passed=true`, rejects conflicting duplicate run keys, and keeps distinct
execution/method configurations in separate aggregate rows.

## 12. Acceptance checklist

A result family is paper-ready only when:

- all repository tests pass;
- BF16 cases E–H pass for every headline model;
- requested and resolved model revisions match immutable commits;
- the GPU, driver, package environment, Git commit, and clean state are recorded;
- every dataset record is hash-verified by the correct frozen manifest;
- no prompt is truncated and no generation is numerically invalid or degenerate;
- every requested sample/method/budget/configuration row completed;
- exact declared FullKV/custom parity coverage passed;
- every compressed row has zero overflow, at least 99% budget utilization, and
  at most one token of shortfall;
- FullKV and compressed scores are paired under the same execution contract;
- the official full GSM8K table uses 1,319 examples, not the 200-example pilot;
- local baseline ports are labeled approximations unless separately replaced
  and validated against official implementations.
