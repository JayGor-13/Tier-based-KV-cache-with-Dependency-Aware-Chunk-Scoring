# Experiment runbook

For the current GSM8K-first workflow, use
[`GSM8K_QUICKSTART.md`](GSM8K_QUICKSTART.md).

The required order is:

1. Native uncompressed GSM8K answer validation.
2. FullKV parity against native generation.
3. Small TDC-KV retention sweep.
4. Scale GSM8K sample count.
5. Only then add more datasets, baselines, ablations, and paper reporting.

Do not use results from a later step when an earlier gate is failing.
