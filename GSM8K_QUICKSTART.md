# GSM8K quick start

For the staged Qwen2.5-7B-Instruct experiment on a Kaggle T4, including 4-bit
loading and checkpoint/resume commands, see [KAGGLE_T4_GSM8K.md](KAGGLE_T4_GSM8K.md).

This is the only workflow to use until GSM8K is stable.

## What we measure

The primary quality metric is **GSM8K final numeric exact-match accuracy**:

```text
accuracy = correct final numeric answers / tested questions
```

Also record:

- parse rate: fraction of outputs containing a final numeric answer;
- end-to-end latency per sample;
- decode throughput in generated tokens per second;
- actual KV retention (`kept KV tokens / original KV tokens`).

`0.3` retention means keeping 30% of the prompt KV cache, or removing about
70%. We call it retention to avoid confusing “compression ratio” conventions.

## 1. Verify the code

From the repository root:

```powershell
.venv\Scripts\python.exe -m pytest
```

If the virtual environment does not exist, follow the setup section in
[`README.md`](README.md).

## 2. Prove that the model can answer GSM8K

Run five questions with native Hugging Face generation:

```powershell
.venv\Scripts\python.exe scripts\test_gsm8k_answers.py --model Qwen/Qwen2.5-1.5B-Instruct --samples 5 --prompt direct
```

This command does **not** use TDC-KV, eviction, a custom KV cache, or attention
scoring. It prints each question, raw model output, parsed answer, gold answer,
correctness, and latency. It saves JSON to
`outputs/gsm8k/native_answers.json`.

The gate passes when:

- at least one output is parseable;
- at least one answer is correct, so accuracy is non-zero;
- outputs are non-empty and sensible when manually inspected;
- no prompt was truncated.

The command exits non-zero when no answer is parseable or accuracy is zero.
That is a useful failure signal, not a reason to start compression testing.

## 3. Test the ChunkKV prompt separately

The prompt implemented in this repository is the **eight-shot** GSM8K prompt
from the cited ChunkKV appendix:

```powershell
.venv\Scripts\python.exe scripts\test_gsm8k_answers.py --model Qwen/Qwen2.5-1.5B-Instruct --samples 5 --prompt chunkkv8 --output outputs/gsm8k/native_answers_chunkkv8.json
```

It is not a 50-shot prompt. A 50-shot version is not currently implemented and
would be a different experimental protocol. Never compare a direct-prompt run
with an eight-shot run in the same accuracy table.

If `direct` works but `chunkkv8` fails, the problem is prompt/model interaction
or truncation—not KV compression. Inspect the saved raw outputs first.

## 4. Run the first compression test

Use exactly the same model and prompt that passed the native test:

```powershell
.venv\Scripts\python.exe scripts\run_gsm8k_compression.py --model Qwen/Qwen2.5-1.5B-Instruct --samples 5 --prompt direct --retention-ratios 0.5,0.3
```

This runs:

- unpruned FullKV;
- TDC-KV at 50% retention;
- TDC-KV at 30% retention;
- native-versus-custom FullKV parity checks.

It prints a small table with parse rate, accuracy, total latency, and decode
tokens/second. The complete result is saved to
`outputs/gsm8k/compression.json`. The command fails if native and custom FullKV
generation do not match. The JSON records the effective repetition penalty,
full token-history scope, EOS token IDs, and stopping rule applied identically
to native, FullKV, and compressed decoding.

## 5. Scale slowly

Only increase one dimension at a time:

```powershell
# Confirm repeatability on 20 questions.
.venv\Scripts\python.exe scripts\test_gsm8k_answers.py --samples 20 --prompt direct --output outputs/gsm8k/native_answers_20.json

# Then run the two initial retention settings on 20 questions.
.venv\Scripts\python.exe scripts\run_gsm8k_compression.py --samples 20 --prompt direct --retention-ratios 0.5,0.3 --output outputs/gsm8k/compression_20.json

# Only after that succeeds, add harder retention settings.
.venv\Scripts\python.exe scripts\run_gsm8k_compression.py --samples 20 --prompt direct --retention-ratios 0.5,0.3,0.2,0.1 --output outputs/gsm8k/compression_20_all_ratios.json
```

Use `--resume` to continue an interrupted compression run with the same command
and output path.

## Model precision

Start with the 1.5B model because it isolates correctness cheaply. `--dtype
auto` preserves the checkpoint's native dtype and the result records the actual
parameter dtype.

An unquantized 8B BF16 model needs roughly 16 GB just for weights, before KV
cache and attention memory. It will not fit a 6 GB GPU. A 4-bit 8B model may fit,
but quantization changes the experiment and the current compression runner has
not qualified that path. Do not use quantization to hide a broken native-answer
or FullKV-parity test.

## Stop conditions

Stop and inspect the JSON rather than scaling when any of these occurs:

- empty output or zero parse rate;
- zero native accuracy;
- prompt truncation;
- NaN/Inf numerical-integrity error;
- custom FullKV differs from native generation;
- missing FullKV or retention groups;
- TDC-KV produces no valid answer at every tested retention.

After the 20-question GSM8K run passes these gates, increase to 100 examples.
Only after the GSM8K curve is stable should you add HotPotQA, NIAH, more
baselines, or ablations.
