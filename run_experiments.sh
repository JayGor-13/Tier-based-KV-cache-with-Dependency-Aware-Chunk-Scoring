#!/bin/bash
set -e

# Setup environment if needed
# source venv/bin/activate

echo "Running Baseline Benchmarks (Offline Traces)..."
python scripts/run_baselines.py \
    --trace-path data/traces/sample_trace.json \
    --methods "chunkkv,snapkv,h2o" \
    --budget 128 \
    --output outputs/baselines_results.json

echo "Running Main TDC-KV Results (Offline Traces)..."
python scripts/run_main_results.py \
    --trace-path data/traces/sample_trace.json \
    --budget 128 \
    --output outputs/tdc_results.json

echo "Running HF Grid Evaluation (Live Model Generation)..."
python scripts/run_hf_grid.py \
    --models "Qwen/Qwen2.5-0.5B-Instruct" \
    --datasets "source=yahma/alpaca-cleaned,split=train,prompt_field=instruction,answer_field=output;source=gsm8k,config=main,split=test,prompt_field=question,answer_field=answer" \
    --budget-ratios "0.5,0.7" \
    --thetas "0.3" \
    --alphas "0.6" \
    --max-samples 5 \
    --max-new-tokens 20 \
    --device auto \
    --output outputs/hf_grid_results.json

echo "All experiments finished!"
