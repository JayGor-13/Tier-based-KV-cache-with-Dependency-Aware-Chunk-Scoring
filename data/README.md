# Trace Data Layout

This repository's experiment runners default to these trace paths:

- `data/niah_trace.jsonl`
- `data/gsm8k_trace.jsonl`
- `data/2wiki_trace.jsonl`
- `data/hotpotqa_trace.jsonl`
- `data/musique_trace.jsonl`

If you don't have these files yet, use `--trace-overrides` in runner scripts to point to local trace files (for example the files in `tmp/`).

## Supported trace formats

- `.jsonl` (one JSON object per line)
- `.json` (single object/list or object containing `samples`)
- `.pt` / `.pth` (list of dicts or dict containing `samples`)

## Minimum fields by mode

### Standard mode (`run_tdc_policy` / baselines)

Required per sample:

- `id` (optional; auto-generated if missing)
- `chunks` (list of token-index lists)
- `chunk_scores` (list of floats, one score per chunk)
- one of:
  - `k_cache` and `v_cache` tensors/arrays, or
  - `k_cache_shape` / `v_cache_shape` / `cache_shape` / enough shape hints (`sequence_length`, `num_heads`, `head_dim`)

Optional:

- `budget`
- `attention_obs` (recommended for SnapKV/H2O realism)
- `gold`, `prediction`

### Full pipeline mode (`--full-pipeline`)

In addition to standard fields, also required:

- `token_ids`
- `attention_obs`
- punctuation ids via either:
  - sample field `punct_ids`, or
  - CLI flag `--punct-ids`

## Example run with overrides

```bash
python scripts/run_experiments.py \
  --benchmarks niah,gsm8k \
  --methods tdc_kv,chunkkv,snapkv,h2o \
  --trace-overrides niah=tmp/sample_trace.jsonl,gsm8k=tmp/sample_trace.jsonl \
  --budget 10 \
  --output outputs/experiments_results.json
```
