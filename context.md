# TDC-KV Repo Context

Last maintained: 2026-08-16

## Purpose

This repository implements a trace-driven prototype of tier-based KV-cache
eviction with dependency-aware chunk scoring. The design details live in
`specifications.md`; common setup and smoke commands live in `README.md`.

## Implemented Pipeline

1. Module 1, `src/core/chunker.py`
   - Builds sentence/punctuation-boundary chunks from token ids.
   - Splits punctuation-free or merged spans at `max_chunk_tokens=64`.
   - Returns ordered chunk tensors and a token-to-chunk map.
   - Supports a one-token incremental append path with the same size cap.

2. Module 2, `src/core/scorer.py`
   - Computes attention-mass signal `M` from observed attention `[H,w,t]`.
   - Routes current relevance through a sparse historical chunk-dependency graph.
   - Exposes direct-attention, dependency-routing, and fused chunk scores.
   - Aggregates token scores to chunks, min-max normalizes, and fuses with
     `alpha=0.6`, `beta=0.4`.
   - Also supports multi-layer attention `[L,H,w,t]` with optional layer weights.

3. Module 3, `src/core/masker.py`
   - Assigns dependency-ranked chunk tiers: `0` eviction candidate, `1` soft-protected,
     `2` hard-protected.
   - Hard-protects the sink chunk and the recent-window chunk range.

4. Module 4, `src/core/evictor.py`
   - Removes whole chunks by ascending score from Tier 0 first, then Tier 1.
   - Boundary-refines at most one selected chunk by removing its oldest
     positions, preventing whole-chunk budget underfill.
   - Never removes Tier 2 unless `allow_level2_fallback=True`.
   - Selects cache positions along sequence axis `-2`, so both `[H,t,d]` and
     batched cache layouts are supported.
   - `src/models/cache_manager.py` keeps logical positions and chunk metadata
     aligned with the compacted cache and re-evicts after every decode step.
   - Decode-time trimming guarantees the post-step cache does not exceed the
     configured budget and uses the same final-group boundary refinement.

5. Benchmarks and baselines
   - `benchmarks/pipeline.py` runs TDC-KV and baseline policies from precomputed
     trace records.
   - `scripts/run_main_results.py`, `scripts/run_baselines.py`,
     `scripts/run_ablations.py`, and `scripts/run_hf_grid.py` are the main
     experiment entry points.
   - Baselines implemented: ChunkKV, SnapKV, and H2O-style observed-attention
     heavy hitters.

6. Phase-1 paper qualification layer
   - Uses one-time raw/chat prompt serialization and records prompt/token hashes.
   - Includes the ChunkKV-compatible GSM8K eight-shot protocol and numeric judge.
   - Checks exact FullKV/custom-cache token parity before qualified experiments.
   - Uses a shared controlled prefill for FullKV/compressed comparisons and
     separates method-specific scoring, policy, and decode timing; CUDA runs
     record synchronized peak memory and physical KV tensor bytes.
   - Uses precomputed H2O/SnapKV scores, independent direct-attention ChunkKV
     scores, and a common decode policy for fair method comparisons.
   - Records Python/package/CUDA/GPU/Git provenance, immutable model/dataset
     revisions, and an environment-bound deterministic grid fingerprint.
   - Writes atomic per-run checkpoints and resumes only a matching experiment.
   - Rejects prompt truncation, incomplete method/budget coverage, non-exact
     NIAH, dirty Git state, failed target-model preflight, and unpinned inputs.

7. Full paper experiment layer
   - `scripts/run_paper_suite.py` defines qualification, tuning, a deterministic
     final-quality pass, three timing repetitions, ablations, and combined
     profiles with model/dataset/sample sharding.
   - The combined profile selects the best macro-dataset tuning configuration,
     freezes it in `selected_config.json`, and applies it to later jobs.
   - Qualification/tuning/final samples come from disjoint frozen record-hash
     partitions; NIAH contexts are exact in each target tokenizer.
   - NIAH and HotpotQA rows include evidence token localization, evidence/chunk
     survival, a global-mask evidence-token eviction proxy, evidence depth, and
     head consensus. No true head/layer GER or layer allocation is claimed.
   - Ablations cover fixed/token chunks, attention/routing balance, Tier 1,
     sink/recent protection, and uniform/linear all-layer scoring.
   - `scripts/preflight_hf_models.py` verifies Hub access and freezes immutable
     model commits; loaded-model checks gate context, sliding cache, CUDA, and VRAM.
   - `scripts/generate_paper_artifacts.py` validates qualified inputs and writes
     aggregate CSV/JSON, Markdown tables, paired tests with Holm correction,
     bootstrap confidence intervals, and paper figures.

## Fixed Issues

- Sparse graph rows are causal query-to-key edges. Routing now propagates
  current relevance backward to historical dependencies with a direct-score
  residual, rather than boosting a recent query from an old key's score.
- The all-layer uniform ablation now assigns exactly equal layer weights;
  linearly increasing weights remain a separate hypothesis.
- Module 2 now moves observed attention and chunk indices to the scorer device
  before computation/indexing.
- Module 2 now rejects negative and out-of-range chunk indices instead of
  allowing PyTorch negative indexing to silently score the wrong token.
- Module 2 incremental score updates now resize to the current number of chunks
  and ignore negative update indices.
- Module 1 no longer requires `transformers` at import time. It only needs
  `transformers` when the tokenizer-loading CLI path is used.
- Module 1 incremental updates normalize `chunk_map` and chunk tensors to the
  constructor device before appending.
- Module 1 bounds every semantic chunk with configurable `max_chunk_tokens`,
  preventing punctuation-free contexts from causing extreme budget underfill.
- Module 4 now meets matched token budgets through a single partial boundary
  chunk after applying normal tier and chunk-score ordering.
- Module 4 now normalizes `mask_tiers` to the `chunk_scores` device in direct
  `compute_keep_mask` calls.
- H2O and SnapKV baselines now trim lowest-scored forced keeps when sink/recent
  protections exceed the requested budget, preserving the budget invariant.
- HuggingFace cache extraction avoids boolean checks on tensors, which can
  raise in PyTorch.
- `transformers` model loading now prefers the current `dtype` keyword and
  falls back to legacy `torch_dtype` for older installs.
- Generated `tmp/` and `outputs/` artifacts were removed from version control
  and are now ignored. The smoke trace was moved to `data/sample_trace.jsonl`.

## Verification

Commands run successfully:

```powershell
python -m pytest
python scripts\run_main_results.py --trace-path data\sample_trace.jsonl --recent-window 4 --output outputs\codex_main_check.json
python scripts\run_baselines.py --trace-path data\sample_trace.jsonl --recent-window 4 --output outputs\codex_baselines_check.json
python scripts\run_ablations.py --trace-path data\sample_trace.jsonl --theta-grid 0.3 --recent-window-grid 4 --output outputs\codex_ablations_check.json
python -c "import src.core; import src.baselines; print('imports ok')"
```

Temporary `outputs/codex_*_check.json` files from verification can be removed
after smoke checks.

## Known Problems and Deferred Improvements

- Live HuggingFace generation with an evicted cache has offline end-to-end
  coverage for tiny GPT-2, Llama, and Qwen2 models, including compressed-logit
  parity and global cache positions. Large checkpoints and additional
  `transformers` releases still require smoke validation because cache APIs are
  model- and version-sensitive.
- `src/models/hf_cache_adapter.py` normalizes legacy tuple caches, Transformers
  v4 `key_cache/value_cache`, and Transformers v5 layer-based caches. Prefill
  extraction, decode compaction, and cache reconstruction share this adapter.
- The prefill evictor can report an over-budget Tier-2-protected result. Before
  generation, the decoding cache manager applies a logged Tier-2 fallback and
  enforces the strict budget; both pre-manager and effective sizes are logged.
- Matched-budget paths validate at least 99% utilization, at most one token of
  shortfall, and zero overflow. Disabling Tier-2 fallback is an explicit
  protection stress test and may intentionally violate the matched budget.
- HuggingFace prefill processes bounded query blocks with a cumulative KV cache.
  Each block is consumed online into the sparse graph and final observation
  window, so dense attention retention is bounded by the configured block size.
  The full prompt KV cache is still required before eviction.
- Module 1 incremental updates do not apply the `min_chunk_tokens` merge rule
  on every append. Aligning incremental and full chunk construction would need a
  deliberate state-management decision.
- Layer weighting in Module 2 is implemented as a scoring hypothesis. The spec
  notes that the empirical validation is still pending.
- The local StreamingLLM, H2O, SnapKV, and ChunkKV policies are fair,
  matched-budget approximations, not vendored official reference repositories.
- The ordinary FP16/BF16 eager-attention loader is intentionally the paper
  correctness path. It does not make 7B/8B models fit the laptop RTX 4050 6 GB;
  large-model qualification and final runs require a verified A100-class Colab
  runtime. Quantized execution would be a separate experimental condition.
