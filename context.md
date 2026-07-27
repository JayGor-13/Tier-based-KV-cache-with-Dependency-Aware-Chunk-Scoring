# TDC-KV Repo Context

Last maintained: 2026-07-27

## Purpose

This repository implements a trace-driven prototype of tier-based KV-cache
eviction with dependency-aware chunk scoring. The design details live in
`specifications.md`; `README.md` currently only contains the project title.

## Implemented Pipeline

1. Module 1, `src/core/chunker.py`
   - Builds sentence/punctuation-boundary chunks from token ids.
   - Returns ordered chunk tensors and a token-to-chunk map.
   - Supports a one-token incremental append path.

2. Module 2, `src/core/scorer.py`
   - Computes attention-mass signal `M` from observed attention `[H,w,t]`.
   - Computes forward-routing signal `R` for the observed query window.
   - Aggregates token scores to chunks, min-max normalizes, and fuses with
     `alpha=0.6`, `beta=0.4`.
   - Also supports multi-layer attention `[L,H,w,t]` with optional layer weights.

3. Module 3, `src/core/masker.py`
   - Assigns chunk tiers: `0` eviction candidate, `1` soft-protected,
     `2` hard-protected.
   - Hard-protects the sink chunk and the recent-window chunk range.

4. Module 4, `src/core/evictor.py`
   - Removes whole chunks by ascending score from Tier 0 first, then Tier 1.
   - Never removes Tier 2 unless `allow_level2_fallback=True`.
   - Selects cache positions along sequence axis `-2`, so both `[H,t,d]` and
     batched cache layouts are supported.

5. Benchmarks and baselines
   - `benchmarks/pipeline.py` runs TDC-KV and baseline policies from precomputed
     trace records.
   - `scripts/run_main_results.py`, `scripts/run_baselines.py`, and
     `scripts/run_ablations.py` are the main trace-driven entry points.
   - Baselines implemented: ChunkKV, SnapKV, and H2O-style observed-attention
     heavy hitters.

## Fixed Issues

- Module 2 forward routing over-boosted recent tokens under uniform attention
  because `R_window` summed over heads while the out-of-window fallback used
  `R=M`. Routing is now normalized by head count so uniform attention produces
  uniform chunk scores.
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
- Module 4 now normalizes `mask_tiers` to the `chunk_scores` device in direct
  `compute_keep_mask` calls.
- H2O and SnapKV baselines now trim lowest-scored forced keeps when sink/recent
  protections exceed the requested budget, preserving the budget invariant.

## Verification

Commands run successfully:

```powershell
python -m pytest -q
python scripts\run_main_results.py --trace-path tmp\sample_trace.jsonl --recent-window 4 --output tmp\codex_main_check.json
python scripts\run_baselines.py --trace-path tmp\sample_trace.jsonl --recent-window 4 --output tmp\codex_baselines_check.json
python scripts\run_ablations.py --trace-path tmp\sample_trace.jsonl --theta-grid 0.3 --recent-window-grid 4 --output tmp\codex_ablations_check.json
python -c "import src.core; import src.baselines; print('imports ok')"
```

Temporary `tmp/codex_*_check.json` files from verification were removed after
the smoke checks.

## Known Problems and Deferred Improvements

- `src/models/cache_utils.py`, `src/models/modeling_llama.py`, and
  `src/models/modeling_phi3.py` are empty. A live transformer integration is
  not implemented in this repo yet.
- `setup.py` and `requirements.txt` are empty. `environment.yml` contains the
  currently necessary runtime/test dependencies for the implemented trace path.
- The main TDC-KV method can fail if hard-protected Tier 2 tokens alone exceed
  the budget. This is consistent with the current hard-protection methodology;
  changing it would require an explicit policy choice, such as reducing the
  recent window or enabling Level-2 fallback.
- Whole-chunk eviction can undershoot or overshoot the exact budget because it
  removes entire chunks. This follows the current chunk-level eviction design.
- Module 1 incremental updates do not apply the `min_chunk_tokens` merge rule
  on every append. Aligning incremental and full chunk construction would need a
  deliberate state-management decision.
- Layer weighting in Module 2 is implemented as a scoring hypothesis. The spec
  notes that the empirical validation is still pending.
