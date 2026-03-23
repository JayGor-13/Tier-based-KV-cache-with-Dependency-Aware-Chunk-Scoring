# Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring

## What is implemented

- Module 1: `src/core/chunker.py`
- Module 2: `src/core/scorer.py`
- Module 3: `src/core/masker.py`
- Module 4: `src/core/evictor.py`
- Full 4-module orchestrator: `src/core/pipeline.py`

The full pipeline now runs:

1. chunk construction from token ids
2. dual-signal chunk scoring from observed attention
3. tier assignment (0/1/2)
4. tier-priority KV eviction

## Running tests

```bash
pytest -q
```

## Unified experiment control (our model + baselines + benchmark selection)

Use:

```bash
python scripts/run_experiments.py \
  --benchmarks niah,gsm8k \
  --methods tdc_kv,chunkkv,snapkv,h2o \
  --trace-overrides niah=tmp/sample_trace.jsonl,gsm8k=tmp/sample_trace.jsonl \
  --budget 10 \
  --output tmp/experiments_results.json
```

Latency benchmarking with warmup (30 runs, discard first 5):

```bash
python scripts/run_experiments.py \
  --benchmarks niah \
  --methods tdc_kv,chunkkv,snapkv,h2o \
  --trace-overrides niah=tmp/sample_trace.jsonl \
  --budget 10 \
  --repeats 30 \
  --warmup-runs 5 \
  --allow-level2-fallback \
  --output outputs/experiments_results.json
```

### Important flags

- `--benchmarks`: `niah,gsm8k,2wiki,hotpotqa,musique` or `all`
- `--methods`: `tdc_kv,chunkkv,snapkv,h2o`
- `--full-pipeline`: for `tdc_kv`, recompute chunks+scores from `token_ids` + `attention_obs`
- `--punct-ids`: required for full-pipeline mode if trace does not include `punct_ids`

## Graphs and plots

Generate PNG comparison plots from result JSONs:

```bash
python scripts/plot_results.py --output-dir outputs/plots
```

Or from unified experiment JSON:

```bash
python scripts/plot_results.py \
  --experiments-json outputs/experiments_results.json \
  --output-dir outputs/plots
```

Latency sweep (synthetic sequence lengths: 512, 2048, 8192):

```bash
python scripts/run_latency_sweep.py \
  --sequence-lengths 512,2048,8192 \
  --methods tdc_kv,chunkkv,snapkv,h2o \
  --repeats 30 \
  --warmup-runs 5 \
  --allow-level2-fallback \
  --plot \
  --output outputs/latency_sweep.json
```

Or plot existing sweep JSON:

```bash
python scripts/plot_results.py \
  --latency-sweep-json outputs/latency_sweep.json \
  --output-dir outputs/plots
```

## Llama / Phi-3 integration

Model wrappers are provided in:

- `src/models/modeling_llama.py` (`LlamaTDCKVModel`)
- `src/models/modeling_phi3.py` (`Phi3TDCKVModel`)

Both wrappers support prefill-time cache compression with:

- `tdc_kv` (our model),
- `chunkkv`,
- `snapkv`,
- `h2o`.

### Example (Llama)

```python
from src.models.modeling_llama import LlamaTDCKVModel

runner = LlamaTDCKVModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
out = runner.prefill_and_compress(
    prompt="Explain the water cycle in 3 steps.",
    budget=1024,
    method="tdc_kv",
    window_size=16,
)
print(out.kept_indices.shape[0])
```

### Example (Phi-3)

```python
from src.models.modeling_phi3 import Phi3TDCKVModel

runner = Phi3TDCKVModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
out = runner.prefill_and_compress(
    prompt="Summarize this document.",
    budget=1024,
    method="h2o",
)
print(out.removed_indices.shape[0])
```
