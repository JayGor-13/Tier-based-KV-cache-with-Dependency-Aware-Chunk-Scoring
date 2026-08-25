# Paper-Comparable TDC-KV Implementation Plan

Status: implementation-ready plan  
Scope: recover numerical correctness, make GSM8K results comparable to prior
work, and validate the dependency-aware contribution on suitable multi-hop
tasks.

## 1. Decision

Do not run another parameter sweep or a full 8B experiment yet. The existing
GSM8K sweep is invalid because every decoded answer is the same repeated token,
the generated token IDs are all zero, and the scoring tensors contain `NaN`.
FullKV also scores zero, so the result does not measure KV-cache compression.

An unquantized 7B/8B model loaded explicitly in BF16 is the intended headline
configuration, but changing model size or dtype alone is not sufficient. The
pipeline must first pass numerical, generation-parity, protocol, and aggregation
gates. The corrupted FP16 checkpoint must never be resumed or included in a
paper table.

The primary direct-comparison result will be:

```text
NQR@30% = 100 * Accuracy(method, actual retention ~= 0.30)
                 / Accuracy(FullKV, same model, samples, prompt, and decoder)
```

Always report NQR with absolute GSM8K accuracy, a 95% confidence interval,
actual KV retention, peak KV/VRAM use, latency, and decode throughput. Do not
collapse quality and efficiency into one composite score.

## 2. What failed and why it matters

| Finding | Consequence | Required response |
|---|---|---|
| All current predictions are 96 repeated `!` characters and all generated IDs are zero. | Task quality is undefined; the zero accuracy is not evidence against TDC-KV. | Reject non-finite and degenerate generations before writing a successful row. |
| Attention/dependency score ranges are `NaN`. | Tier selection and eviction order are meaningless. | Fail at the first non-finite tensor and record the exact stage/block/layer. |
| FullKV is also zero. | Compression is not the source of the observed quality failure. | Establish native-HF versus custom-cache FullKV parity before testing compression. |
| Paper and local suites hard-code FP16 and last-layer attention. | FP16 may trigger the failure; the configured layer weighting is inactive. | Make dtype and attention collection explicit protocol fields; qualify BF16. |
| Qualification accepts non-empty token lists without checking tensor health or output degeneracy. | Broken runs can pass the current gate. | Add numerical-health, parity, parse, finish, and truncation requirements. |
| Summary grouping omits experiment profile and major protocol fields. | Main, sweep, or differently configured runs can be merged. | Aggregate only identical protocol fingerprints and de-duplicate by `run_key`. |
| Local baseline implementations are approximations. | They cannot support strong “beats paper X” claims. | Use official/reference implementations for headline comparisons or label results as approximations. |
| Current 1.5B/1.7B models and 200-sample pilot do not match the cited GSM8K tables. | Cross-paper accuracy numbers are not directly comparable. | Reproduce exact model/prompt/budget settings and use all 1,319 GSM8K test items for the final direct table. |
| The current prompt is short and decoding uses a Python loop. | Present throughput numbers do not establish long-context speedups. | Benchmark long inputs separately with controlled kernels, warm-up, and stage timings. |
| Last-layer global-mask TDC-KV is not the same as the proposal's all-layer formulation or true head/layer GER. | The implementation and novelty claim can diverge. | Select and name a canonical variant through an ablation; narrow claims to what is implemented. |

## 3. Non-negotiable experiment rules

1. Every compressed result is paired with FullKV using the same model revision,
   sample IDs, serialized prompt tokens, dtype, attention backend, decoding
   policy, maximum lengths, and seed.
2. “30%” means 30% **retention** (70% compression). Store both values and the
   measured post-eviction retention to avoid paper terminology ambiguity.
3. Tune TDC-KV only on frozen training/development data: GSM8K train, HotpotQA
   train, and separate synthetic NIAH seeds. Never tune on the official GSM8K
   test records or LongBench final records, even if they are placed in a
   disjoint local manifest.
4. Use greedy decoding for the direct GSM8K reproduction unless the reproduced
   paper's released protocol specifies otherwise.
5. Use unquantized BF16 for every method in the headline run. A quantized 8B
   model may be an additional deployment study, not a direct substitute.
6. Keep invalid, interrupted, pilot, tuning, timing, ablation, and final rows in
   separate artifacts. A report must refuse to merge incompatible fingerprints.
7. Report local baseline ports as `approximate` until they pass conformance
   tests against their official/reference repositories.
8. Use an explicit, versioned prompt serialization. `auto` serialization is not
   permitted in a direct reproduction because library/chat-template changes can
   alter the prompt.

Maintain four separate protocol fingerprints:

- `chunkkv_gsm8k_replication`: exact released models, Appendix-style eight-shot
  prompt, full GSM8K test set, and published budgets.
- `chunkkv_longbench_replication`: official LongBench data, prompts,
  truncation, generation limits, and evaluators.
- `tdc_multihop_controlled`: BF16 matched-method experiment for the TDC claim.
- `efficiency_benchmark`: fixed hardware and long-context systems protocol.

The reproduction track answers “can we compare to the published table?” The
controlled track answers “under identical local conditions, does TDC-KV help?”
Do not merge or average the two tracks.

## 4. Delivery sequence and gates

```text
P0 numerical guardrails
  -> G0 finite forward/generation
  -> P1 native/custom FullKV qualification matrix
  -> G1 exact generation parity
  -> P2 protocol fingerprint + safe aggregation
  -> G2 reproducible 10-sample pilot
  -> P3 official baseline/model reproduction
  -> G3 all-method pilot at 30% retention
  -> P4 full GSM8K direct comparison
  -> G4 frozen 1,319-sample result
  -> P5 multi-hop and long-context validation
  -> G5 claim-ready evidence
```

No phase may consume final-test compute until its incoming gate passes.

## 5. Phase P0 — quarantine and numerical guardrails

### Changes

- Treat the current CSV/checkpoint as read-only failure evidence. Start all new
  runs under a new output root such as `outputs/v2_bf16_qualified/`; do not use
  `--resume` against an old root.
- Add `scripts/validate_results.py` to inspect legacy files without modifying
  them. It must flag non-finite values, missing provenance/parity, degenerate
  generation, duplicate/conflicting keys, and mixed protocols. Schema-v1 files
  are `legacy_unqualified` and cannot be promoted into a paper artifact. The
  supplied 216-row checkpoint must fail all-successful-row validation.
- In `src/models/cache_utils.py`, add a reusable finite-value validator for:
  model logits, per-layer attentions, every returned KV tensor, concatenated
  blockwise KV, scorer inputs, and decode-step logits/KV. Include stage,
  prefill-block index, layer, dtype, device, shape, finite count, minimum,
  maximum, and absolute maximum in an exception/diagnostic record.
- Check the first prefill block, every subsequent block, the first selected
  token, and every decode step. Never apply `argmax` to non-finite logits.
- In `src/core/scorer.py` and `src/core/dependency_graph.py`, reject non-finite
  inputs and intermediates. Do not allow min/max normalization to propagate
  `NaN` or silently convert it into an eviction score.
- In `src/core/masker.py`, remove the behavior that lets invalid scores continue
  after replacing them with a sentinel. Non-finite ranking input is a failed
  run, not a low-priority chunk.
- Apply the same finite-input contract in `src/core/evictor.py`,
  `src/baselines/_utils.py`, and `src/models/cache_manager.py` so no alternate
  method path can conceal corruption.
- Add a `numerical_health` object to every result row with, at minimum:
  `passed`, `first_failure_stage`, `prefill_blocks_checked`,
  `decode_steps_checked`, `nonfinite_tensor_count`, and observed dtypes.
- Write failed rows/checkpoints with an explicit failure status and traceback,
  but exclude them from quality aggregation.
- Serialize raw rows and checkpoints with `allow_nan=False` and atomic temporary
  file replacement. Bump the result schema/validation version so an old
  permissive checkpoint cannot resume into a qualified run.

### Tests

- `tests/test_scorer.py`: finite input succeeds; injected `NaN`/`Inf` fails with
  the responsible component named.
- `tests/test_hf_prefill_integration.py`: single-block and multi-block prefill
  produce finite logits, attentions, and KV on the smallest supported model.
- `tests/test_hf_cache_e2e.py`: injected non-finite prefill/decode tensors stop
  the run before token selection.
- `tests/test_masker.py`: non-finite scores cannot be ranked or evicted.

### Gate G0

- Zero non-finite tensors in 10 native-HF and 10 custom-cache samples.
- Zero degenerate repeated-token outputs caused by a constant invalid token ID.
- Any injected non-finite tensor creates a failed row and a useful diagnostic.

## 6. Phase P1 — isolate FP16, blockwise, and cache correctness

### Diagnostic matrix

First run three real frozen GSM8K prompts on the existing 1.5B model, then run
the same 10-prompt qualification on each headline model with identical greedy
settings:

| Case | Path | Dtype | Prefill | Purpose |
|---|---|---|---|---|
| A | Native Hugging Face FullKV | FP16 | native | Establish whether the model/device is stable in FP16. |
| B | Direct eager forward | FP16 | whole prompt | Isolate eager-attention numerical behavior. |
| C | Custom unpruned FullKV | FP16 | one block | Test cache reconstruction without block concatenation. |
| D | Custom unpruned FullKV | FP16 | block sizes 128 and 16 | Isolate mask/position/cache errors in blockwise prefill. |
| E | Native Hugging Face FullKV | BF16 | native | Headline reference path. |
| F | Direct eager forward | BF16 | whole prompt | Establish a finite eager-attention reference. |
| G | Custom unpruned FullKV | BF16 | one block | Validate BF16 cache reconstruction. |
| H | Custom unpruned FullKV | BF16 | block sizes 128 and 16 | Validate block-size invariance. |

Interpretation:

- A fails: model/device/backend FP16 is unsafe; use BF16 and retain the failure
  as a documented environment limitation.
- A passes but B fails: investigate eager-attention/FP16 behavior.
- B and C pass, D fails: fix blockwise attention mask, cache position, position
  IDs, cache mutation, or concatenation; dtype is not the primary cause.
- The FP16 cases fail and E-H pass: FP16 range is the leading cause, but retain
  the diagnostic rather than assuming model size caused it.
- E-H disagree: fix the custom cache/generation implementation before any
  compression experiment.

### Changes

- Remove hard-coded `float16` from `scripts/run_local_paper_suite.py` and
  `scripts/run_paper_suite.py`. Expose dtype as a required recorded protocol
  choice; default the paper profile to `bfloat16` only after model/device
  preflight confirms support.
- Make `auto` preserve or select a documented model-native safe dtype rather
  than automatically mapping every CUDA run to FP16.
- Make the attention backend strict. If eager attention is requested and the
  loader fallback drops that request, fail preflight rather than silently
  changing the protocol.
- Extend `benchmarks/model_preflight.py` to record the requested dtype, actual
  parameter dtype set, model-config dtype, resolved attention backend, GPU and
  compute capability, BF16 support, and the absence of 4/8-bit quantization.
- Add a qualification profile/script that executes the matrix and emits a
  compact comparison report with prompt-token equality, generated-token
  equality, first differing step, max logit difference, and numerical health.
- Increase GSM8K `max_new_tokens` to 512 for direct qualification. Calibrate and
  freeze the final value only after inspecting the FullKV output-length
  distribution; any clipping triggers a protocol-wide rerun, not a per-sample
  extension.
- Extend `benchmarks/qualification.py` to require numerical health, required
  parity coverage for the declared target-model/sample set, generation-health
  fields, parse-rate threshold, and zero prompt/output truncation for the direct
  profile. Recompute rather than trust stored summaries; validate runtime,
  cache-byte, and budget arithmetic without crashing on malformed fields.
- Record decoded-nonempty status, unique-token count, maximum-token fraction,
  EOS/length termination, and answer parseability. Reject max-length sequences
  dominated by one non-special token; do not reject token ID zero merely because
  zero is a valid vocabulary ID.

### Gate G1

- Native BF16 FullKV and custom unpruned BF16 FullKV have identical prompt
  tokens and identical greedy generated token IDs on all 10 samples.
- All required parity records exist; none are optional or missing.
- At least 9/10 GSM8K qualification outputs contain a parseable final number,
  with zero prompt/output truncation and no repeated-token degeneration. In the
  final task metric, a legitimate unparseable answer remains in the denominator
  as incorrect; it is never silently excluded.
- The selected 7B/8B model passes a one-sample memory preflight on the target GPU.
- Requested BF16 is confirmed as actual BF16 parameters on BF16-capable hardware,
  eager attention is confirmed, and no quantization is active.

## 7. Phase P2 — result schema, provenance, and aggregation

### Schema and completion contract

Add `benchmarks/result_schema.py` with schema version 2 and explicit validation.
Every result set must record:

- `state`: `in_progress`, `complete`, or `failed`,
- job and protocol fingerprints,
- requested and resolved model/tokenizer revisions, dtype, device, and backend,
- code/environment provenance and actual ordered sample-manifest hash,
- planned, attempted, successful, failed, and schema-validated run counts.

A row becomes `status=ok` only after numerical and row-schema validation. Paper
qualification is always recomputed from raw rows; a stored `passed=true` flag is
never trusted. Keep separate `diagnostic` and `paper` qualification levels so a
small smoke check cannot be mislabeled as paper-qualified.

### Canonical protocol fingerprint

Compute a stable hash over the following canonical fields and write the fields
plus hash into every row/checkpoint/report:

- Git commit and dirty-state marker; code/schema version.
- Model ID, exact model/tokenizer revisions, dtype, device, attention backend.
- Dataset ID/config/revision, manifest hash, partition, sample ID/order.
- Protocol ID/version, prompt serialization mode, exact prompt-token hash.
- Maximum prompt length, maximum new tokens, stop/decoding policy, seed.
- Prefill block size, cache implementation, method implementation source/version.
- Experiment profile (`qualification`, `tuning`, `main`, `timing`, `ablation`).
- Method, requested retention, actual retention, and every active method parameter.
- TDC variant, attention layer mode, layer weighting, chunker configuration.

Implement these identities in `benchmarks/experiment_identity.py`:

- `execution_contract_id`: resolved model/data/protocol/code/environment fields,
- `sample_input_id`: record hash, rendered-prompt hash, and input-token hash,
- `method_config_id`: implementation revision, physical budget, method
  parameters, attention/chunk/decode policy,
- `run_key`: the three IDs plus seed and repetition.

The job fingerprint additionally includes the exact ordered manifest, requested
grid, shard definition, and environment. Calculate it only after records are
materialized and hashed, not from sample IDs alone.

### Checkpoint and resume contract

Replace the repeatedly rewritten permissive JSON checkpoint with a standard
library SQLite store:

- metadata table for schema and the immutable job contract,
- runs table keyed by `run_key`,
- attempt-history table retaining failures/retries,
- parity table keyed by model/dataset/sample/control configuration.

Only validated successful rows enter the resumable completed set. Resolve and
verify actual model revision, dtype, and backend before binding the store. A
changed record hash, manifest, revision, dtype, backend, protocol, generation
limit, code revision, method grid, or selected configuration is a hard resume
mismatch. Export final JSON atomically only after exact planned-run coverage.

### Aggregation rules

- Group only by a complete protocol fingerprint plus the intended comparison
  axis. Refuse mixed fingerprints with a clear list of differing fields.
- De-duplicate by `run_key`. Identical duplicates count once; conflicting
  duplicates are a hard error.
- Exclude execution-corrupt/unqualified rows from task metrics while reporting
  their count and reasons. Never turn an execution failure into a score of zero.
  By contrast, a valid but wrong or unparseable task response counts as an
  incorrect GSM8K prediction.
- Pair each compressed sample with its matching FullKV row before computing
  NQR or paired statistics. Missing pairs are a qualification failure.
- Produce separate `pilot`, `tuning`, `main`, `timing`, and `ablation` summaries.
- Paper production accepts only the explicit output list in the suite manifest;
  remove recursive `*.json` discovery. An `--allow-unqualified` diagnostic mode
  must label its output visibly and can never create a paper artifact.
- Add GSM8K accuracy with Wilson 95% intervals. For method comparisons, report
  paired bootstrap confidence intervals and a paired binary test (McNemar);
  apply Holm correction when testing multiple methods/budgets.
- Report actual-retention distribution, not only the requested target.

### Tests

- `tests/test_local_paper_summary.py`: profile/dtype/protocol differences never
  merge; default parameter rows from different experiments remain separate.
- `tests/test_result_aggregation.py`: duplicate and conflicting `run_key`
  behavior; failed-row exclusion; exact FullKV pairing.
- `tests/test_paper_reporting.py`: NQR, Wilson interval, paired interval/test,
  actual-retention columns, and missing-pair rejection.
- `tests/test_reproducibility.py`: fingerprint stability and sensitivity to each
  protocol field.
- `tests/test_result_schema.py` and `tests/test_experiment_identity.py`: strict
  JSON/schema validation, identity sensitivity, actual-record hashing, and
  immutable job-contract behavior.
- `tests/test_hf_cache_e2e.py`: interrupt/resume equivalence, retry history,
  changed-contract rejection, and exact planned coverage.

### Gate G2

- Re-running a 10-sample FullKV/TDC pilot into a fresh output root gives the same
  run keys, predictions, metrics, and fingerprint.
- Deliberately changing dtype, profile, model revision, prompt serialization,
  or generation length produces a different fingerprint and cannot merge.
- The summary has exactly the expected unique sample-method-budget pairs.
- An interrupted/resumed run produces the same semantic rows as an uninterrupted
  run, while changing one contract field or a record's content rejects resume.

## 8. Phase P3 — baseline fidelity and canonical TDC-KV

### Models and baselines

Use exact released revisions after recording license/access approval:

- Direct-reproduction models: Qwen2-7B-Instruct and Llama-3-8B-Instruct variants
  used by the target ChunkKV table.
- Optional reasoning extension: DeepSeek-R1-Distill-Llama-8B.
- Required methods: FullKV, StreamingLLM, H2O, SnapKV, ChunkKV, TDC-KV, plus
  random matched-budget retention as a sanity control.
- PyramidKV: include an official/reference implementation if integration and
  conformance finish; otherwise cite its reported result as contextual only.
- KVpop: contextual comparison unless evaluated with its own model, task,
  sampling, and teacher-relative protocol. Do not compare its number directly
  to a GSM8K greedy-accuracy result.

For each baseline, record `implementation_source`, commit, local patch, and
whether it is `official`, `reference`, or `approximate`. Test retention and
selected token indices on small fixtures against reference outputs.
Match actual physical K/V bytes summed across all layers, not only a nominal
token ratio; this is required for fair comparison to unequal per-layer policies
such as PyramidKV. Provide both the method-native decode policy and a separate
common-policy isolation ablation where relevant.

### Select the TDC variant

Run a tuning-only ablation comparing:

1. last-layer scoring (the current production behavior),
2. all-layer uniform scoring,
3. all-layer linearly weighted scoring (the theoretical formulation), and
4. any memory-reduced approximation proposed for deployment.

The all-layer weighted formulation is the intended headline implementation;
validate it against a dense mathematical oracle on small inputs. Last-layer
scoring remains an ablation. If all-layer scoring cannot be implemented, update
the algorithm definition and claims before running final experiments rather
than presenting last-layer output under the all-layer method description. Do
not claim PyramidKV-style per-layer allocation or true head/layer GER unless it
is actually implemented.

### Tune without test leakage

- Coarse stage: 50-100 frozen samples from GSM8K train and HotpotQA train plus
  separate NIAH seeds.
- Confirmation: the top three configurations on 200 training/development
  samples per natural dataset plus 100 NIAH cases.
- Tune at 30% and 10% measured retention over `alpha`, `theta`, recent window,
  dependency top-k, chunk limit, and the active layer-mode ablation.
- Select on mean NQR across tuning datasets and retentions; require NIAH NQR of
  at least 95%, then break ties by lower method-specific latency.
- Select one shared configuration on Llama-3-8B and transfer it unchanged to
  Qwen2/Mistral unless a separately labeled model-specific-tuning study is run.
- Freeze the chosen configuration, ranking, tuning record hashes, and protocol
  fingerprint before any final test generation.

### Gate G3

On 10 frozen final-like samples per model at 30% retention:

- Every method passes numerical and output qualification.
- Every compressed row has a FullKV pair and measured retention within a
  predeclared tolerance (target: +/- 1 percentage point, or document why token
  granularity makes that impossible).
- ChunkKV uses its paper setting, including fixed chunk size 10 for the direct
  reproduction. TDC-KV may use its frozen semantic chunker, but the distinction
  must be labeled.
- Official/reference baseline outputs pass their conformance checks.
- No final parameters are changed after this gate.

## 9. Phase P4 — paper-comparable GSM8K experiment

### Frozen protocol

- Dataset: all 1,319 GSM8K test examples.
- Prompt: the frozen eight-shot chain-of-thought protocol already represented in
  `benchmarks/gsm8k_protocol.py`; verify exact tokenization per model.
- Models: Qwen2-7B-Instruct and Llama-3-8B-Instruct target revisions.
- Precision/backend: unquantized BF16, eager attention, same for all methods.
- Decoder: greedy, batch size 1, deterministic seed, and the exact released
  generation limit when verified. Use `max_new_tokens=512` provisionally; freeze
  the value after FullKV calibration and rerun the whole protocol if any output
  is clipped.
- Direct table retention points: 30%, 20%, and 10%. Also keep 100% FullKV once
  per model/sample. Additional 50%, 25%, and 12.5% points belong in a separate
  curve and must not be presented as exact table reproduction.
- Run three deterministic repetitions only if the reproduced protocol changes
  anything across repetitions. If greedy runs are byte-identical, report that
  fact and do not use duplicate rows as independent samples.

### Staged compute

1. Complete the 10-sample all-method pilot.
2. Run 100 samples and inspect qualification, paired deltas, output lengths, and
   actual retention without tuning anything.
3. Run the remaining 1,219 samples exactly once under the frozen fingerprint.
4. Generate final artifacts only after sample-count and pairing audits pass.

The minimum direct table reports, for every model/method/retention:

- exact numeric-answer accuracy and correct/total,
- Wilson 95% interval,
- NQR versus paired FullKV,
- paired accuracy delta with interval and corrected significance result,
- requested and actual retention/compression,
- failure, parse, finish, and truncation rates.

Before interpreting compression, compare local FullKV to the cited reproduction
targets (approximately 71.1% for Qwen2-7B and 76.8% for Llama-3-8B in the target
table). A predeclared discrepancy larger than three percentage points triggers
a model-revision, prompt-wrapping, decoding, and evaluator audit; it is not
explained away as method variance.

### Gate G4

- Exactly 1,319 unique, qualified sample pairs per reported configuration.
- Zero missing FullKV pairs, conflicting duplicates, non-finite rows, or prompt
  truncations; output truncation below the predeclared threshold (target zero).
- Dataset manifest, prompt-token hashes, revisions, and fingerprint are frozen.
- Paper table can be regenerated from raw rows by one deterministic command.

## 10. Phase P5 — validate the actual dependency-aware contribution

GSM8K supports direct comparison but is weak evidence for bridge-token
preservation. Add official LongBench adapters and run their official prompts,
truncation, maximum generation settings, evaluators, and complete task sets
(normally 200 examples per task) for:

- HotpotQA distractor validation,
- 2WikiMultiHopQA,
- MuSiQue,
- Needle-in-a-Haystack as a retrieval control rather than a reasoning task.

For multi-hop data, report official answer EM/F1 plus, where annotations permit,
supporting-fact recall and an evidence/bridge survival metric measured after
eviction. Freeze a tuning/test split and match actual retention across methods.
State whether pruning is question-aware: if the question appears in the prompt
used to score chunks, label TDC-KV as question-aware and compare only against
equally informed baselines or discuss the advantage explicitly.

For the ChunkKV LongBench reproduction, run Llama-3-8B at 30%, 20%, and 10%
retention and Qwen2-7B/Mistral-7B-v0.3 at the published comparison points. Keep
standalone HotpotQA validation separate from LongBench HotpotQA. Report the
three-task mean normalized quality (`MH-NQR`) in addition to each task score.
Use at least 500 frozen, evidence-annotated validation examples per native
multi-hop dataset for the separate mechanistic evidence-survival study.

Required ablations:

- no dependency graph,
- no evidence protection,
- no sink/recent hard protection,
- uniform versus dependency-aware chunk score,
- last versus all-layer scoring,
- shared global mask versus any implemented layer-specific allocation,
- sentence chunks versus fixed size 10.

### Gate G5

- TDC-KV improves or preserves paired quality at matched actual retention on at
  least one true multi-hop task, and the relevant ablation removes/reduces the
  gain.
- Evidence/bridge survival moves in the predicted direction.
- Claims use “global shared-mask proxy” unless true per-head/per-layer GER has
  been implemented and tested.

## 11. Phase P6 — efficiency study

Run this independently from the short GSM8K quality table on the same 40 GB+
GPU class (A100/A40 or equivalent) for every method:

- batch size 1; identical model, dtype, backend, kernels, and decoding loop,
- tokenizer-exact 4K and 8K input lengths; at least 1,024 generated tokens where
  the model/task supports it,
- five warm-up runs followed by at least 20 steady-state repetitions in
  randomized method order,
- synchronize the device around timed regions,
- measure prefill, scoring/graph, eviction, decode, and end-to-end time,
- report median, p95, bootstrap interval, tokens/s, time-to-first-token, peak
  allocated and reserved VRAM, measured KV bytes, and OOM/failure rate.
- include 4K input/1K output, 8K/1K, and 8K/4K configurations; accept a timing
  series when coefficient of variation is below 5% or document and rerun an
  unstable configuration.

The current approximately 777-token prompt and custom Python decode loop cannot
support a headline speed claim. Quality and systems comparisons must use the
same correctness-qualified implementation; optimized kernels can be an
additional labeled path after parity testing.

## 12. File-level implementation backlog

| Priority | Files | Work item | Done when |
|---|---|---|---|
| P0 | `scripts/validate_results.py` | Read-only legacy audit and quarantine classification | All 216 corrupt `ok` rows are rejected and no legacy file can become a paper artifact. |
| P0 | `benchmarks/numerical_validation.py` | Structured tensor-health exception/helper | Every failure carries stage/tensor/block/layer context. |
| P0 | `src/models/cache_utils.py` | Tensor-health checks and block/decode diagnostics | Invalid tensors stop before token selection and identify the first bad stage. |
| P0 | `src/core/scorer.py`, `src/core/dependency_graph.py`, `src/core/masker.py` | Enforce finite scoring contract | No non-finite score can reach eviction. |
| P0 | `benchmarks/qualification.py` | Numerical, parity, parse, degeneracy, finish, truncation gates | Corrupted checkpoint pattern is rejected. |
| P0 | `scripts/run_paper_suite.py`, `scripts/run_local_paper_suite.py`, `paper_environment.json` | Remove hard-coded FP16; add diagnostic profile and BF16 preflight | A-H matrix is reproducible from CLI and fully recorded. |
| P0 | `benchmarks/model_preflight.py` | Enforce actual dtype/backend/hardware/quantization contract | Requested BF16/eager settings are verified, not merely logged. |
| P0 | HF/cache/scorer/qualification tests | Regression coverage for the observed failure | Tests fail on zero-token/NaN fixture and pass on qualified BF16 fixture. |
| P1 | `benchmarks/result_schema.py`, `benchmarks/experiment_identity.py`, `benchmarks/reproducibility.py` | Schema-v2 identity and canonical protocol fingerprint | Every row is traceable to code, data, prompt, model, and protocol. |
| P1 | runner/checkpoint storage | Transactional SQLite attempts/runs/parity store and atomic final export | Resume accepts only the identical verified contract and exact coverage. |
| P1 | `scripts/summarize_local_paper_results.py`, `scripts/aggregate_hf_results.py` | Safe grouping, de-duplication, pairing, invalid-row accounting | Mixed profiles/dtypes cannot merge. |
| P1 | `benchmarks/paper_reporting.py`, `benchmarks/eval_metrics.py` | NQR, Wilson CI, paired inference, actual-retention reporting | Unit-tested direct table is generated from paired raw rows. |
| P1 | `src/baselines/*`, `benchmarks/hf_runner.py` | Official/reference integration metadata and conformance fixtures | Each headline baseline is versioned and classified. |
| P1 | `benchmarks/gsm8k_protocol.py`, manifests | Freeze exact direct-reproduction protocol and 1,319 IDs | Prompt tokens and sample order are hash-stable. |
| P1 | tuning/experiment scripts | Canonical TDC layer-mode ablation and frozen selection | Main variant matches the written method definition. |
| P1 | dataset manifests/tuning | Move tuning off final/test records | Train/dev hashes are frozen and final records are unseen. |
| P2 | multi-hop runners/metrics | HotpotQA, 2Wiki, MuSiQue evidence-aware evaluation | Official quality plus evidence-survival metrics are reproducible. |
| P2 | runtime metrics/reporting | Long-context synchronized benchmark harness | Stage and end-to-end efficiency statistics are comparable. |
| P2 | README, runbook, paper artifacts | Replace FP16 and mixed-profile instructions; document claims | A new researcher can reproduce every accepted table. |

## 13. Required test/CI matrix

Run fast CPU tests on every change, then a small GPU qualification before merge:

| Layer | Required coverage |
|---|---|
| Unit | finite validator, scorer normalization, masker contract, schema/identity/fingerprint, NQR/CI, dedupe/pairing |
| Integration | native/custom FullKV parity; one-block/multi-block prefill; transactional resume with matching contract |
| Negative | injected `NaN`/`Inf`, repeated zero token, missing parity, parse failure, prompt/output truncation, mixed dtype/profile, conflicting duplicate |
| GPU smoke | one small model in FP16 and BF16; one target 7B/8B model in BF16; FullKV and TDC at one budget |
| Reproduction | 10 fixed GSM8K samples across all headline methods at 30% retention |

CI must not require an 8B model, but final-run qualification must.

## 14. Reporting and claim policy

Use three separate evidence layers:

1. **Direct reproduction:** exact GSM8K models, eight-shot prompt, and 30/20/10%
   retention; compare absolute accuracy and NQR with ChunkKV/reference baselines.
2. **Algorithm validation:** multi-hop quality, bridge/evidence survival, and
   ablations demonstrating the dependency component.
3. **Systems evaluation:** matched long-context latency, throughput, and memory.

Allowed claim examples after all gates pass:

- “At 30% measured retention, TDC-KV retains X% of paired FullKV GSM8K
  accuracy on model Y.”
- “TDC-KV improves paired HotpotQA F1 by X over baseline Z at matched retention,
  while increasing annotated evidence survival by Y.”

Do not claim that TDC-KV beats a paper from unmatched models/tasks, that
approximate baseline ports reproduce official methods, that BF16 itself fixes
the algorithm, or that short-prompt Python-loop timings prove deployment speed.

## 15. Immediate execution order

1. Run the read-only legacy validator, classify the bad CSV/checkpoint as invalid
   evidence, and create a fresh v2 output namespace.
2. Implement P0 finite-value/schema checks, strict atomic serialization, and
   their negative tests.
3. Implement the A-H diagnostic matrix. Start with all cases on three 1.5B
   prompts, then repeat the passing
   BF16 paths on 10 target-model prompts.
4. Fix blockwise/cache logic if BF16 custom FullKV does not exactly match native
   BF16 FullKV.
5. Harden identity, transactional resume, qualification, aggregation, and
   fingerprinting; prove the observed bad checkpoint is rejected and mixed
   profiles cannot merge.
6. Validate all-layer TDC against the mathematical oracle, integrate official
   baselines, tune only on training/dev data, and freeze the canonical variant.
7. Run the 10-sample, 7B/8B BF16, all-method pilot at 30% retention.
8. Run the 100-sample checkpoint, then the remaining full GSM8K test set.
9. Run multi-hop ablations and the independent long-context efficiency study.
10. Generate tables/plots only from qualified, paired, fingerprint-consistent
    rows and review every claim against the evidence layer that supports it.

The first full-method 8B run is therefore step 7, after correctness and protocol
identity are proved; only small target-model qualification runs occur earlier.
