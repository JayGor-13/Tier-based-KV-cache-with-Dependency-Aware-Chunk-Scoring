# TDC-KV

Tier-based KV-cache compression with dependency-aware chunk scoring.

The current milestone is deliberately narrow: make GSM8K reliable first. Do
not begin multi-dataset sweeps or ablations until the native model produces
parseable answers and non-zero GSM8K accuracy.

## Start here

Follow [`GSM8K_QUICKSTART.md`](GSM8K_QUICKSTART.md). The two commands are:

```powershell
python scripts/test_gsm8k_answers.py --samples 5 --prompt direct
python scripts/run_gsm8k_compression.py --samples 5 --prompt direct --retention-ratios 0.5,0.3
```

The first command uses native Hugging Face generation with no custom cache
code. The second compares FullKV and TDC-KV on the same GSM8K records and prompt.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -c constraints-paper.txt
pip install -e .
```

Use `HF_TOKEN` only when the selected Hugging Face model requires it.

## Test the repository

```powershell
python -m pytest
python -c "import src.core; import src.baselines; print('imports ok')"
```

Offline tests use tiny random models, so most code can be verified without
downloading a model checkpoint.

## Main code

- `scripts/test_gsm8k_answers.py`: native, uncompressed answer validation.
- `scripts/run_gsm8k_compression.py`: small FullKV-versus-TDC-KV experiment.
- `scripts/run_hf_grid.py`: general experiment runner; use it only after the
  GSM8K foundation passes.
- `benchmarks/gsm8k_protocol.py`: direct and ChunkKV eight-shot prompts.
- `benchmarks/eval_metrics.py`: GSM8K final numeric exact-match judge.
- `src/core/`: chunking, dependency scoring, tier assignment, and eviction.
- `src/models/`: Hugging Face prefill and cache-aware decoding.

Generated experiment files belong under `outputs/`, which is ignored by git.
The larger paper workflow remains available in
[`LOCAL_PAPER_SUITE_FLOW.md`](LOCAL_PAPER_SUITE_FLOW.md), but it is not the
starting point.
