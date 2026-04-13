# Repository Audit and Experiment Plan

## Scope and execution checks

- Repository shape: 39 Python files (~4,272 non-comment LOC), with core modules, baselines, benchmark runners, model wrappers, and tests.
- Test status: `pytest -q` passes (`30 passed`).
- Coverage tooling gap: `pytest --cov=...` currently fails because `pytest-cov` is not installed.

## Architecture review

### Strengths

1. **Clear modular decomposition**
   - Core logic is cleanly separated into chunking/scoring/masking/eviction modules with an orchestrator (`src/core/*`, `src/core/pipeline.py`).
2. **Consistent dataclass outputs**
   - `MaskerResult`, `EvictionResult`, and pipeline result dataclasses make outputs explicit and easy to consume.
3. **Baseline comparators are integrated**
   - ChunkKV, SnapKV, H2O share compatible interfaces for fairer method comparisons.
4. **Model-level integration exists for Llama/Phi-3**
   - Wrapper supports prefill + compression for both TDC-KV and baselines.

### Complexity profile (big-O, dominant paths)

- **Chunker (`SentenceBoundaryChunkConstructor.forward`)**: `O(t * |punct_ids|)` for membership mask (vectorized equality against punctuation IDs) + `O(t)` chunk map build.
- **Scorer (`DualSignalScorer.forward`)**:
  - single-layer path: approximately `O(H * w * t)`;
  - multi-layer path: approximately `O(L * H * w * t)`;
  - chunk aggregation: `O(t)` total over chunks.
- **Masker (`assign_protection_tiers`)**: quantile + chunk lookup, approximately `O(M log M)` (quantile backend dependent) + chunk scans.
- **Evictor (`compute_keep_mask`)**: chunk sort by score per tier + index marking, roughly `O(M log M + t)`.

## Critical issues and risks

1. **Packaging is incomplete (`setup.py` empty)**
   - `setup.py` is currently blank, so standard package build/install workflow is effectively missing.
   - Impact: brittle reproducibility for users expecting `pip install -e .` with project metadata.

2. **Default experiment paths point to missing files**
   - `scripts/run_experiments.py` defaults to `data/*.jsonl`, but this repo currently has no `data/` directory.
   - Impact: defaults fail out-of-the-box unless `--trace-overrides` (or `--trace-dir`) is supplied.

3. **Chunker punctuation vocab build is expensive and exception-swallowing**
   - `build_punctuation_vocab` decodes every tokenizer ID and catches broad exceptions.
   - Impact: slow startup on large vocabs; hidden decode failures can silently reduce punctuation coverage.

4. **Input validation can be stronger in scorer path**
   - `DualSignalScorer._validate_chunks` only checks non-empty chunk list, but not index bounds, monotonicity, or overlap.
   - Impact: malformed chunk inputs can produce hard-to-debug downstream indexing errors.

5. **Failure handling in experiment scripts can hide root causes**
   - `run_experiments.py` and `run_latency_sweep.py` catch broad exceptions in per-sample loops.
   - Impact: pipelines can appear to run while silently accumulating failed samples, reducing trust in summaries.

6. **No automated lint/type/static checks in baseline workflow**
   - Current test suite validates behavior, but no enforced style/type checks are configured in this repo.
   - Impact: regressions in interface assumptions can slip in despite unit tests.

## Observed correctness state

- Core unit/integration tests currently pass for chunker/scorer/masker/evictor/pipeline/baselines (`30 passed`).
- This indicates the current implementation is internally consistent for tested scenarios, but does **not** guarantee correctness under large-model runtime stress (memory pressure, long-context extremes, mixed precision edge cases).

## What to do next (prioritized)

1. **Fix project hygiene first (high priority)**
   - Add real `setup.py` or migrate fully to `pyproject.toml`.
   - Add `pytest-cov` and at least one static checker (ruff or mypy).
   - Add `data/README.md` explaining trace schema + expected files.

2. **Add robustness checks (high priority)**
   - Strengthen chunk validation in scorer/evictor entry points.
   - Add explicit logging for failed samples in experiment scripts and summary counters by failure reason.

3. **Prepare reproducible local model experiments (high priority)**
   - Start with Phi-3 Mini locally (resource-feasible).
   - Run Llama only if hardware is sufficient, otherwise use quantization / smaller variants.

4. **Run a staged experiment plan (high priority)**
   - Stage A: synthetic + trace-based correctness/latency checks.
   - Stage B: real-model prefill compression sanity on a small prompt set.
   - Stage C: benchmark-scale run with repeated latency and warmup.

## Detailed local experiment steps for Llama and Phi-3

### 0) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional but recommended for coverage:

```bash
pip install pytest-cov
```

### 1) Quick health checks

```bash
pytest -q
python scripts/run_experiments.py \
  --benchmarks niah,gsm8k \
  --methods tdc_kv,chunkkv,snapkv,h2o \
  --trace-overrides niah=tmp/sample_trace.jsonl,gsm8k=tmp/sample_trace.jsonl \
  --budget 10 \
  --output outputs/experiments_results.json
```

### 2) Local model smoke test (Phi-3 first)

Use this Python snippet:

```python
from src.models.modeling_phi3 import Phi3TDCKVModel

runner = Phi3TDCKVModel.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto",
    device_map="auto",
)
out = runner.prefill_and_compress(
    prompt="Summarize this article in 3 bullets.",
    budget=1024,
    method="tdc_kv",   # also try: chunkkv, snapkv, h2o
    window_size=16,
    recent_window=16,
)
print("kept:", out.kept_indices.numel(), "removed:", out.removed_indices.numel())
```

### 3) Llama local smoke test

```python
from src.models.modeling_llama import LlamaTDCKVModel

runner = LlamaTDCKVModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
out = runner.prefill_and_compress(
    prompt="Explain the KV cache in simple terms.",
    budget=1024,
    method="tdc_kv",
    window_size=16,
)
print(out.kept_indices.shape[0])
```

### 4) If local hardware is limited

- Prefer **Phi-3 mini** first.
- Reduce prompt length and budget for smoke tests.
- If OOM happens, use smaller models, lower context, or quantized inference stack.

### 5) Benchmark execution recipe

```bash
python scripts/run_experiments.py \
  --benchmarks niah,gsm8k,2wiki,hotpotqa,musique \
  --methods tdc_kv,chunkkv,snapkv,h2o \
  --trace-overrides \
niah=tmp/bench_niah.json,gsm8k=tmp/bench_gsm8k.json,2wiki=tmp/bench_2wiki.json,hotpotqa=tmp/bench_hotpotqa.json,musique=tmp/bench_musique.json \
  --budget 1024 \
  --repeats 30 \
  --warmup-runs 5 \
  --allow-level2-fallback \
  --output outputs/experiments_results.json
```

Then plot:

```bash
python scripts/plot_results.py \
  --experiments-json outputs/experiments_results.json \
  --output-dir outputs/plots
```

### 6) Real-model evaluation protocol (recommended)

- Keep prompts fixed across methods.
- For each prompt/model pair, run all methods with the same budget.
- Capture:
  - kept/removed token counts,
  - prefill+compress latency,
  - downstream generation quality metric (task-specific).
- Repeat at least 5 times and report mean/std.

## Suggested immediate backlog

1. Add packaging metadata and installable project config.
2. Add `pytest-cov` and enforce minimum coverage threshold.
3. Add validation assertions for chunk index correctness.
4. Add explicit failed-sample diagnostics in JSON output.
5. Add a reproducible `data/` trace preparation script.
