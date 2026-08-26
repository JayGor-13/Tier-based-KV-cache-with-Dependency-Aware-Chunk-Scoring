# Kaggle T4 GSM8K: Clone, Set Up, and Run

This is the complete copy/paste runbook for moving the current repository from
Windows to Kaggle and running the GSM8K experiment progressively.

The target experiment uses:

- model: `Qwen/Qwen2.5-7B-Instruct`;
- weights: bitsandbytes 4-bit NF4;
- compute: FP16 (`W4A16` overall);
- prompt: repository ChunkKV eight-shot GSM8K prompt;
- serialization: the Qwen chat template;
- attention backend: eager, as required by the attention-score collection;
- decoding: deterministic greedy decoding;
- comparison: paired FullKV versus TDC-KV on the same examples.

These results can compare TDC-KV with its paired FullKV control. Report the
model as `W4A16/NF4`; do not describe it as an FP16-weight reproduction.

## Important before cloning

`git clone` downloads only commits that exist on GitHub. It does **not** copy
uncommitted files from the Windows working directory.

The Kaggle-preparation changes are currently on the local branch `branch-h`.
Commit and push them before opening Kaggle. The commands below deliberately do
not add `.pytest_tmp_qualification_fix/`, which is a local temporary test
directory and should not be pushed.

## Part A — Publish the current code from Windows

Run these steps in Windows PowerShell, outside Kaggle.

### 1. Open the repository

```powershell
cd "C:\Users\jaygo\Desktop\DESKTOP\RMS\Code\Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring"
```

### 2. Confirm the branch, remote, and changed files

Run each command separately:

```powershell
git branch --show-current
git remote -v
git status --short
```

Expected branch:

```text
branch-h
```

Expected remote:

```text
https://github.com/JayGor-13/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring.git
```

Read the `git status` output before continuing. Do not stage
`.pytest_tmp_qualification_fix/`.

### 3. Stage only the prepared source, tests, and documentation

Copy this as one PowerShell command:

```powershell
git add -- GSM8K_QUICKSTART.md KAGGLE_T4_GSM8K.md benchmarks/hf_runner.py benchmarks/model_preflight.py pyproject.toml scripts/run_gsm8k_compression.py scripts/run_hf_grid.py scripts/test_gsm8k_answers.py src/models/cache_utils.py tests/test_gsm8k_foundation_scripts.py tests/test_model_loading.py tests/test_model_preflight.py
```

Check exactly what is staged:

```powershell
git status --short
git diff --cached --stat
```

The temporary `.pytest_tmp_qualification_fix/` directory may still appear as
untracked. That is expected; leave it untracked.

### 4. Commit the changes

```powershell
git commit -m "Prepare Kaggle T4 GSM8K experiments"
```

If Git says that your identity is unknown, set it for this repository and then
repeat the commit:

```powershell
git config user.name "YOUR GITHUB NAME"
git config user.email "YOUR GITHUB EMAIL"
git commit -m "Prepare Kaggle T4 GSM8K experiments"
```

### 5. Push `branch-h` to GitHub

```powershell
git push -u origin branch-h
```

If GitHub asks you to authenticate, complete the browser or credential-manager
login. Do not put a GitHub password or token inside a source file.

### 6. Record the exact commit that Kaggle must use

```powershell
git rev-parse HEAD
```

Copy the printed 40-character commit hash into a note. Kaggle must print the
same hash later.

### 7. Verify the pushed branch in a browser

Open:

<https://github.com/JayGor-13/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring/tree/branch-h>

Confirm that the page contains `KAGGLE_T4_GSM8K.md` and that the newest commit
is the one just pushed. Do not continue if the GitHub page still shows the old
code.

## Part B — Create and configure the Kaggle notebook

### 8. Create a notebook

1. Sign in at <https://www.kaggle.com/>.
2. Select **Create** and then **New Notebook**.
3. Keep the notebook language set to Python.
4. Give the notebook a useful name, for example
   `tdc-kv-qwen25-7b-gsm8k`.

### 9. Enable the GPU and internet

1. Open the notebook **Settings** or **Session options** panel.
2. Set **Accelerator** to **GPU T4 x2** when that option is available.
3. Turn **Internet** on. Internet is required for `git clone`, package
   installation, the Hugging Face model, and the GSM8K dataset.
4. Accept the session restart if Kaggle requests one after changing the
   accelerator.

The current runner uses `cuda:0`, so it uses the first T4. A T4 x2 Kaggle
session does not automatically split this model across both GPUs.

### 10. Verify the assigned GPU

Create the first Kaggle code cell and run:

```python
!pwd
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Continue only if at least one row identifies an NVIDIA T4. If CUDA is absent,
reopen the accelerator setting and restart the session.

### 11. Clone `branch-h` into `/kaggle/working`

Create a new Kaggle code cell and run exactly:

```python
%cd /kaggle/working
!git clone --branch branch-h --single-branch https://github.com/JayGor-13/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring.git
%cd /kaggle/working/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring
```

The persistent notebook working copy is now:

```text
/kaggle/working/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring
```

Do not use the Windows `.venv\Scripts\python.exe` path on Kaggle. Kaggle is
Linux; use `python` in every experiment command.

### 12. Verify that Kaggle cloned the correct branch and commit

Run:

```python
!git branch --show-current
!git rev-parse HEAD
!git status --short
!test -f KAGGLE_T4_GSM8K.md && echo "Kaggle runbook found"
```

Required results:

- the branch is `branch-h`;
- the commit hash exactly matches the Windows hash from step 6;
- `git status --short` prints nothing;
- the final line says `Kaggle runbook found`.

If the hashes differ, do not run the experiment. Fix the push or clone first.

### 13. Install only the missing runtime packages

Run this before importing `transformers`, `datasets`, or `bitsandbytes`:

```python
!python -m pip install "bitsandbytes>=0.45,<1" "transformers>=4.43,<6" "accelerate>=1.14,<2" "datasets>=5,<6" "huggingface-hub>=0.36,<1"
```

Do not reinstall PyTorch: Kaggle already supplies a CUDA-enabled PyTorch build.
Also do not run `pip install -e .` in this notebook. The scripts add the
repository root to Python's import path directly, while an editable install can
be blocked if Kaggle's Python version is older than the version declared by the
project.

If the notebook imported any of these libraries before installation, restart
the Python kernel after `pip` finishes, then run the `%cd` command from step 11
again. A kernel restart resets the current directory.

### 14. Verify Python, CUDA, and package versions

Run:

```python
import torch
import transformers
import datasets
import accelerate
import bitsandbytes

print("python torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("gpu 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print(
    "gpu 0 VRAM GiB:",
    round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    if torch.cuda.is_available()
    else None,
)

assert torch.cuda.is_available(), "CUDA is not available"
assert "T4" in torch.cuda.get_device_name(0), "The first GPU is not a T4"
```

### 15. Verify the prepared command-line options

Run:

```python
%cd /kaggle/working/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring
!python scripts/test_gsm8k_answers.py --help
!python scripts/run_gsm8k_compression.py --help
```

The help output must include `--attn-implementation`, `--quantization`,
`--bnb-4bit-compute-dtype`, `--bnb-4bit-quant-type`,
`--bnb-4bit-double-quant`, and `--require-cuda`.

### 16. Create the result directory

```python
!mkdir -p outputs/kaggle_t4
```

## Part C — Run the GSM8K gates in order

Do not begin with 50 or 1,319 samples. Run each gate below and inspect its JSON
before moving to the next one.

Every multiline command below is a complete Kaggle code cell. Keep `%%bash` as
the first line of the cell.

### 17. Gate 1: native one-sample answer test

This tests the uncompressed Hugging Face model, chat serialization, GSM8K
answer parsing, eager attention, and the 4-bit CUDA load.

```bash
%%bash
set -euo pipefail
python scripts/test_gsm8k_answers.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 1 \
  --prompt chunkkv8 \
  --serialization chat \
  --max-new-tokens 512 \
  --device cuda \
  --attn-implementation eager \
  --quantization bnb-4bit \
  --bnb-4bit-compute-dtype float16 \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-double-quant \
  --require-cuda \
  --output outputs/kaggle_t4/native_qwen25_7b_w4a16_1.json
```

Continue only if:

- the model loaded on CUDA;
- the actual attention implementation is eager;
- 4-bit NF4 loading is recorded;
- the answer is non-empty and parseable;
- the prompt was not truncated;
- the script reports nonzero accuracy.

The JSON is saved even if the final nonzero-accuracy gate fails, so inspect the
raw record before diagnosing the model or prompt.

### 18. Gate 2: one-sample FullKV-versus-TDC-KV test

```bash
%%bash
set -euo pipefail
python scripts/run_gsm8k_compression.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 1 \
  --prompt chunkkv8 \
  --serialization chat \
  --max-new-tokens 512 \
  --retention-ratios 0.8 \
  --device cuda \
  --attn-implementation eager \
  --quantization bnb-4bit \
  --bnb-4bit-compute-dtype float16 \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-double-quant \
  --parity-samples 1 \
  --require-cuda \
  --output outputs/kaggle_t4/compression_qwen25_7b_w4a16_1.json
```

Do not continue unless:

- native Hugging Face and custom FullKV text parity pass;
- native Hugging Face and custom FullKV token parity pass;
- CUDA qualification passes;
- generation and numerical health pass;
- actual TDC-KV retention is approximately `0.8`.

### 19. Gate 3: five-sample smoke test

```bash
%%bash
set -euo pipefail
python scripts/run_gsm8k_compression.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 5 \
  --prompt chunkkv8 \
  --serialization chat \
  --max-new-tokens 512 \
  --retention-ratios 0.8 \
  --device cuda \
  --attn-implementation eager \
  --quantization bnb-4bit \
  --bnb-4bit-compute-dtype float16 \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-double-quant \
  --parity-samples 5 \
  --require-cuda \
  --output outputs/kaggle_t4/compression_qwen25_7b_w4a16_5.json
```

Require 100% parse rate, nonzero FullKV accuracy, 5/5 parity, healthy
generation, and no numerical failure. Inspect every raw answer in the JSON
before scaling.

### 20. Gate 4: fifty-sample pilot

This adds a second retention ratio and an incremental SQLite checkpoint.

```bash
%%bash
set -euo pipefail
python scripts/run_gsm8k_compression.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 50 \
  --prompt chunkkv8 \
  --serialization chat \
  --max-new-tokens 512 \
  --retention-ratios 0.8,0.5 \
  --device cuda \
  --attn-implementation eager \
  --quantization bnb-4bit \
  --bnb-4bit-compute-dtype float16 \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-double-quant \
  --parity-samples 5 \
  --require-cuda \
  --checkpoint outputs/kaggle_t4/qwen25_7b_w4a16_50.checkpoint.sqlite \
  --output outputs/kaggle_t4/compression_qwen25_7b_w4a16_50.json
```

To resume the **same** request after a controlled interruption, preserve the
SQLite file and rerun the exact command with `--resume` added. Changing the
model, prompt, seed, ratios, dtype, attention backend, or quantization options
invalidates the checkpoint; start a new checkpoint for a changed experiment.

### 21. Scale only after the pilot passes

Run 200 samples before the full GSM8K test set. For the full run, use:

- `--samples 1319`;
- `--retention-ratios 0.8,0.5,0.3`;
- `--parity-samples 5`;
- a new descriptive `--checkpoint` path;
- a new descriptive `--output` path.

Before the final experiment, copy the immutable model and dataset revisions
recorded by the successful pilot into `--model-revision` and
`--dataset-revision`. Then start a new checkpoint. A full eager-attention 7B
run may require multiple Kaggle sessions.

## Part D — Preserve and download results

### 22. List the generated artifacts

```python
!find outputs/kaggle_t4 -maxdepth 1 -type f -printf "%f  %s bytes\n"
```

At minimum, preserve every JSON result and every SQLite checkpoint.

### 23. Create one downloadable ZIP file

Run:

```python
import shutil

archive = shutil.make_archive(
    "/kaggle/working/tdc_kv_gsm8k_results",
    "zip",
    root_dir=(
        "/kaggle/working/"
        "Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring/"
        "outputs/kaggle_t4"
    ),
)
print(archive)
```

Optionally create a clickable link inside the notebook:

```python
from IPython.display import FileLink

FileLink("/kaggle/working/tdc_kv_gsm8k_results.zip")
```

### 24. Save the Kaggle notebook output

1. Select **Save Version** after the command finishes and the ZIP exists.
2. Confirm that notebook outputs are included in the saved version.
3. Open the notebook's output/files panel.
4. Download `tdc_kv_gsm8k_results.zip`.
5. Also keep the notebook version as the execution record.

Do not rely only on a live Kaggle session. Save or download the JSON and SQLite
files before ending the session.

## Part E — Update an existing Kaggle clone

Use this only when the Kaggle copy has no local changes. Push the new Windows
commit first, then run these commands in Kaggle:

```python
%cd /kaggle/working/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring
!git status --short
!git fetch origin branch-h
!git checkout branch-h
!git pull --ff-only origin branch-h
!git rev-parse HEAD
```

Compare the final hash with `git rev-parse HEAD` on Windows again. If the
Kaggle clone has local changes, keep it for its outputs and make a fresh clone
in a new Kaggle session rather than overwriting those changes.

## Part F — Common failures

### `Repository not found` or authentication failure

- Confirm that internet is enabled in Kaggle.
- Confirm that the GitHub URL opens in a private/incognito browser window.
- Confirm that `branch-h` was pushed.
- If the repository is private, use the private-repository appendix below.

### `Remote branch branch-h not found`

Run this on Windows:

```powershell
git push -u origin branch-h
```

Then start a fresh clone.

### `Destination path ... already exists`

Do not run the clone cell twice. Move into the existing directory:

```python
%cd /kaggle/working/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring
```

Then verify its branch and commit. For a clean second copy, clone to a new
directory name:

```python
%cd /kaggle/working
!git clone --branch branch-h --single-branch https://github.com/JayGor-13/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring.git tdc-kv-fresh
%cd /kaggle/working/tdc-kv-fresh
```

### CUDA is unavailable

The Kaggle accelerator was not enabled or the session did not restart. Enable
the T4 accelerator, restart the session, and repeat the verification cells.

### `bitsandbytes` cannot find CUDA

1. Confirm that `nvidia-smi` works.
2. Confirm that `torch.cuda.is_available()` is `True`.
3. Rerun the package-install cell.
4. Restart the Python kernel.
5. Run `%cd` back into the repository and repeat the version check.

### Prepared command-line flags are missing

Kaggle cloned an old commit or the wrong branch. Compare the Kaggle and Windows
commit hashes. Do not work around missing flags; correct the clone.

### Model or dataset download fails

- Confirm that Kaggle internet is on.
- Rerun the same gate; Hugging Face downloads can resume from their cache.
- Do not change the experiment protocol merely to bypass a transient download
  problem.

## Appendix — Safely clone a private GitHub repository

Skip this appendix if the repository URL opens without signing in.

1. Create a fine-grained GitHub personal access token with read-only access to
   this repository's contents.
2. In the Kaggle notebook, open **Add-ons > Secrets** (the label may appear as
   a Secrets panel in some layouts).
3. Add a secret named `GITHUB_TOKEN` and paste the token there.
4. Never paste the token directly into a notebook cell or clone URL.
5. Use this Kaggle Python cell instead of the public clone cell:

```python
import base64
import subprocess
from kaggle_secrets import UserSecretsClient

repository_url = (
    "https://github.com/JayGor-13/"
    "Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring.git"
)
destination = (
    "/kaggle/working/"
    "Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring"
)

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
basic_value = base64.b64encode(
    f"x-access-token:{token}".encode("utf-8")
).decode("ascii")

subprocess.run(
    [
        "git",
        "-c",
        f"http.extraHeader=AUTHORIZATION: basic {basic_value}",
        "clone",
        "--branch",
        "branch-h",
        "--single-branch",
        repository_url,
        destination,
    ],
    check=True,
)

del token
del basic_value
```

Then run:

```python
%cd /kaggle/working/Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring
!git branch --show-current
!git rev-parse HEAD
```

## Metrics to report

For each actual retention ratio, report:

- FullKV GSM8K final-numeric Exact Match (EM);
- TDC-KV GSM8K final-numeric Exact Match (EM);
- accuracy delta in percentage points;
- accuracy retention (`TDC-KV EM / FullKV EM`);
- paired preservation among FullKV-correct examples;
- parse rate and generation-limit rate;
- actual KV retention and KV bytes saved;
- prefill, scoring, policy, and decode latency separately;
- decode tokens per second;
- GPU, model revision, dataset revision, eager backend, W4A16/NF4, and package
  versions.

## References

- [Kaggle notebook environment and settings](https://www.kaggle.com/docs/notebooks)
- [GitHub remote repositories and HTTPS cloning](https://docs.github.com/en/get-started/git-basics/about-remote-repositories)
