#!/bin/bash
# run_experiments.sh
# Master script to run full benchmarks for TDC-KV and baselines.

set -e

MODELS="Qwen/Qwen2.5-0.5B-Instruct"
# Evaluate on GSM8K (reasoning) and HotpotQA (multi-hop)
DATASETS="source=gsm8k,config=main,split=test,prompt_field=question,answer_field=answer;source=hotpot_qa,config=distractor,split=validation,prompt_field=question,answer_field=answer"

# Compression ratios (Standard MBE Ladders: 50%, 25%, 12.5%, 6.25%)
BUDGET_RATIOS="0.5,0.25,0.125,0.0625"

# TDC-KV Parameters
THETAS="0.2,0.3,0.4"
ALPHAS="0.4,0.6,0.8"
RECENT_WINDOWS="16,32"

MAX_SAMPLES=100
MAX_NEW_TOKENS=256
DEVICE="cuda" # Change to 'cpu' if running locally without GPU

echo "=========================================================="
echo "Starting Full Scale TDC-KV Experiments"
echo "=========================================================="
echo "Models: $MODELS"
echo "Datasets: $DATASETS"
echo "Max Samples per Dataset: $MAX_SAMPLES"
echo "Max New Tokens (for generation): $MAX_NEW_TOKENS"
echo "=========================================================="

# 1. Run TDC-KV Grid Search
echo ""
echo ">>> [1/2] Running TDC-KV Grid Search..."
python scripts/run_hf_grid.py \
    --models "$MODELS" \
    --datasets "$DATASETS" \
    --budget-ratios "$BUDGET_RATIOS" \
    --thetas "$THETAS" \
    --alphas "$ALPHAS" \
    --recent-windows "$RECENT_WINDOWS" \
    --max-samples $MAX_SAMPLES \
    --max-new-tokens $MAX_NEW_TOKENS \
    --device "$DEVICE" \
    --output "outputs/full_tdckv_results.json"

echo "TDC-KV experiments completed! Results saved to outputs/full_tdckv_results.json"

# Note: If you want to add Baseline runs (ChunkKV, SnapKV, H2O), you would add them here.
# e.g., python scripts/run_baselines.py ...

echo "=========================================================="
echo "All experiments finished successfully."
echo "=========================================================="
