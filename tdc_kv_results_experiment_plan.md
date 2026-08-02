# TDC-KV Paper Results And Experiment Plan

This document specifies how to produce the result section for the TDC-KV paper: result metrics, main tables for three datasets, figures, ablation studies, parameter sweeps, baselines, model choices, and implementation gaps to close before running final experiments.

Project inspected:

- Code: `C:\Users\jaygo\Desktop\DESKTOP\RMS\Code\Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring`
- Papers: `C:\Users\jaygo\Desktop\DESKTOP\RMS\Papers`

Papers used:

- ChunkKV: semantic-preserving chunk-level KV cache compression.
- PyramidKV: layerwise pyramidal KV budget allocation based on attention funneling.
- KVpop: learned fixed-budget predictive online pruning using future-attention supervision.
- Understanding the Physics of KV Cache Compression: reachability-aware evaluation, Global Eviction Ratio, head consensus, and safety-cliff analysis.

## 1. Core Claim To Prove

TDC-KV should be evaluated as a training-free, chunk-level, tier-protected KV cache eviction method that preserves semantic and dependency structure better than token-only pruning under matched KV budgets.

The paper should prove four claims:

1. TDC-KV preserves downstream task quality at high KV compression.
2. TDC-KV improves over token-level and chunk-level baselines under the same cache budget.
3. The dual-signal scorer matters: attention mass alone is weaker than attention mass plus forward-routing dependency score.
4. Tier protection matters: hard-protecting sink and recent chunks prevents failure at aggressive compression.

## 2. Main Datasets

Use three primary datasets in the paper. These cover different stress types and map well to the existing codebase.

| Dataset | Why it is needed | Main metric | Secondary metrics |
|---|---|---:|---|
| GSM8K / Many-shot GSM8K | In-context mathematical reasoning, used by ChunkKV | Accuracy, Exact Match | Token F1, latency, memory |
| Needle-In-A-Haystack (NIAH) | Long-context retrieval and positional robustness, used by ChunkKV and PyramidKV | Retrieval Accuracy | Depth-position heatmap score, F1 |
| HotpotQA | Multi-hop reasoning and dependency preservation | Token F1, Exact Match | Answer-support reachability, GER |

Optional extension datasets:

- 2WikiMultihopQA: use as a second multi-hop QA validation dataset.
- MuSiQue: use as a harder compositional multi-hop test.
- LongBench subset: use if time allows, especially NarrativeQA, Qasper, HotpotQA, 2Wiki, MuSiQue, TREC, TriviaQA, and MultiNews.

## 3. Models To Use

Use at least two backbone families because PyramidKV and the Physics paper show that LLaMA-like and Qwen-like models have different layer/routing behavior.

Recommended main models:

| Model | Role | Why |
|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` or local equivalent | Main LLaMA-family result | Used by ChunkKV and PyramidKV-style comparisons. |
| `mistralai/Mistral-7B-Instruct-v0.3` | Cross-architecture robustness | Used in ChunkKV and PyramidKV. |
| `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2-7B-Instruct` | Qwen-family robustness | ChunkKV uses Qwen2; Physics uses Qwen2.5; KVpop uses Qwen3. |

Optional stronger comparison:

| Model | Role | Why |
|---|---|---|
| `Qwen/Qwen3-4B` or `Qwen/Qwen3-8B` | KVpop comparison setting | KVpop reports Qwen3-4B and Qwen3-8B. Use if the environment supports it. |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Reasoning-heavy setting | ChunkKV evaluates this model for GSM8K-style reasoning. |

Minimum publishable setup:

- Main: LLaMA-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct.
- Datasets: GSM8K, NIAH, HotpotQA.
- Budgets: 50%, 25%, 12.5%, 6.25% retention.
- Runs: 3 random seeds or 3 repeated runs, reporting mean and standard deviation.

## 4. Methods To Compare

Use matched-budget evaluation: every method must receive the same effective KV budget.

| Method | Type | Implemented locally? | Paper link/reason |
|---|---|---:|---|
| FullKV | No compression | Needs explicit baseline in tables | Upper-bound quality. |
| StreamingLLM | Sink + recent window | Not currently in `src/baselines`, should add | Standard baseline in ChunkKV, PyramidKV, KVpop. |
| H2O | Attention heavy hitters + recent | Yes | Standard token-level heavy-hitter baseline. |
| SnapKV | Observation-window attention score | Yes | Strong attention-score baseline. |
| ChunkKV | Semantic chunk selection | Yes | Closest chunk-level baseline. |
| PyramidKV | Layerwise budget allocation | Partially represented via layer weights, but not full baseline | Important because TDC-KV uses layer-insight motivation. |
| KVpop | Learned future-utility pruning | Not necessary to reimplement for main paper, compare conceptually or use public model if feasible | Learned upper-bound competitor. |
| TDC-KV | Ours | Yes | Tiered dependency-aware chunk scoring. |

For publication, the critical baselines are FullKV, StreamingLLM, H2O, SnapKV, ChunkKV, and TDC-KV. PyramidKV and KVpop can be included as paper-reported/contextual comparisons unless implementation time allows.

## 5. Metrics To Report

### 5.1 Task Quality Metrics

| Metric | Definition | Datasets |
|---|---|---|
| GSM8K accuracy | Exact equality of the extracted final numeric answer; references use the value after `####` | GSM8K |
| NIAH retrieval accuracy | Fraction of outputs containing the complete normalized needle as a contiguous token sequence | NIAH |
| HotpotQA answer F1 | Official normalized token-overlap F1, including the special `yes`/`no`/`noanswer` rule | HotpotQA |
| HotpotQA answer EM | Official lowercase, punctuation/article-removed exact match | HotpotQA |
| Generic output EM/F1 | Full-generation string diagnostics only; never the headline score for the three main datasets | All |
| Pass@1 | Single deterministic rollout correctness | Optional for AIME/HMMT-style KVpop comparison |
| Rouge-L | LongBench summarization only | Optional |
| Edit similarity | LongBench code tasks only | Optional |

### 5.2 Cache Efficiency Metrics

Already implemented in `benchmarks/eval_metrics.py`:

| Metric | Formula / meaning |
|---|---|
| Original length | Number of prompt tokens before eviction. |
| Budget | Target retained KV tokens. |
| Kept length | Actual retained KV tokens after eviction. |
| Removed length | Original length - kept length. |
| Retention ratio | Kept length / original length. |
| Compression ratio | 1 - retention ratio. |
| Compression multiplier | Original length / kept length. |
| Budget gap | Kept length - budget; zero is expected for matched-budget runs. |
| Budget utilization | Kept length / min(budget, original length). Must be at least 0.99. |
| Budget shortfall | max(target budget - kept length, 0). Must be at most one token. |
| Budget overflow | max(kept length - target budget, 0). Must be zero. |
| Latency ms | Time spent scoring and evicting, or end-to-end generation latency in HF runs. |

Add for final paper:

| Metric | Why needed |
|---|---|
| Peak GPU memory / VRAM | KVpop reports VRAM; this directly proves memory benefit. |
| Tokens/sec or decoding throughput | ChunkKV and KVpop both report efficiency. |
| Prefill time and decode time separately | Avoid hiding cache-policy overhead inside generation. |
| Eviction overhead per trigger | Shows TDC-KV is lightweight. |

### 5.3 Structural Metrics Inspired By The Physics Paper

These are important differentiators. Add them even if only for HotpotQA/NIAH and a smaller subset.

| Metric | Definition | Why it matters |
|---|---|---|
| Answer-token retention | Fraction of gold-answer/evidence tokens retained in the cache. | Shows whether evidence survives. |
| Global Eviction Ratio (GER) | Fraction of answer-relevant tokens evicted across all heads/layers. | Predicts safety cliff and hallucination. |
| Head consensus | Diversity of top-attended tokens across heads per layer. | Detects representational rigidity. |
| Layer retention profile | Retained-token count per layer. | Shows whether TDC-KV behaves like or unlike PyramidKV. |
| Tier distribution | Count/percentage of chunks in Tier 0, Tier 1, Tier 2. | Shows whether masking is doing meaningful work. |
| Evidence chunk survival | Fraction of chunks containing answer/supporting facts retained. | Direct proof of dependency-aware semantic preservation. |

## 6. Main Result Tables

### Table 1: Main Results Across Three Datasets

Purpose: headline result table.

Rows: methods.

Columns:

| Method | GSM8K Acc | NIAH Acc | HotpotQA F1 | Avg Quality | Retention | Compression | Latency ms | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullKV | | | | | 100% | 0% | | |
| StreamingLLM | | | | | 25% | 75% | | |
| H2O | | | | | 25% | 75% | | |
| SnapKV | | | | | 25% | 75% | | |
| ChunkKV | | | | | 25% | 75% | | |
| TDC-KV | | | | | 25% | 75% | | |

Run this table at the main budget: 25% retention / 75% compression. This matches KVpop's 75% compression setting and is also a readable midpoint before the extreme 12.5% and 6.25% budgets.

### Table 2: Matched-Budget Quality Curve

Purpose: show degradation as compression increases.

Rows: methods.

Columns:

| Method | 50% Retention | 25% Retention | 12.5% Retention | 6.25% Retention |
|---|---:|---:|---:|---:|
| FullKV | reference | reference | reference | reference |
| StreamingLLM | | | | |
| H2O | | | | |
| SnapKV | | | | |
| ChunkKV | | | | |
| TDC-KV | | | | |

Create one version per dataset:

- Table 2a: GSM8K Accuracy vs retention.
- Table 2b: NIAH Accuracy vs retention.
- Table 2c: HotpotQA F1 vs retention.

### Table 3: Efficiency Table

Purpose: demonstrate practical value.

Columns:

| Method | Retention | Scoring overhead ms | Eviction overhead ms | Prefill ms | Decode ms/token | Throughput tokens/sec | Peak VRAM GB |
|---|---:|---:|---:|---:|---:|---:|---:|

Run on one representative long dataset, preferably NIAH at 8k/16k context and HotpotQA at long-context prompt length.

### Table 4: Ablation Study

Purpose: prove each component of TDC-KV matters.

Rows:

| Variant | Description |
|---|---|
| TDC-KV full | Sentence chunks + attention mass + forward routing + tiers. |
| No forward routing | `alpha=1.0`, `beta=0.0`; attention mass only. |
| Routing only | `alpha=0.0`, `beta=1.0`. |
| No tiers | Rank chunks only; no Tier 1/2 protection. |
| No sink protection | Remove sink hard-protection. |
| No recent protection | Remove recent-window hard-protection. |
| Token-level scoring | Same score but evict individual tokens instead of chunks. |
| Fixed-size chunks | Replace sentence-boundary chunks with fixed chunks of 8/16 tokens. |
| Uniform layer weights | Disable PyramidKV-inspired layer weighting. |
| Layer-weighted scoring | Use all layers with higher-layer weighting. |

Columns:

| Variant | GSM8K Acc | NIAH Acc | HotpotQA F1 | Retention | GER | Latency ms |
|---|---:|---:|---:|---:|---:|---:|

### Table 5: Parameter Sensitivity

Purpose: justify defaults.

Parameters:

| Parameter | Grid | Default |
|---|---|---:|
| `theta` | 0.1, 0.2, 0.3, 0.4, 0.5 | 0.3 |
| `recent_window` | 8, 16, 32, 64, 128 | 16 |
| `alpha` | 0.0, 0.25, 0.5, 0.6, 0.75, 1.0 | 0.6 |
| `min_chunk_tokens` | 1, 4, 8, 16, 32 | 5 |
| `max_chunk_tokens` | 32, 64, 128 | 64 |
| budget ratio | 0.5, 0.25, 0.125, 0.0625 | 0.25 |
| layer mode | last layer, all layers uniform, all layers weighted | last/all depending final choice |
| `allow_level2_fallback` | false, true | true for matched-budget results |
| minimum budget utilization | 0.95, 0.99, 1.0 | 0.99 |
| maximum budget shortfall | 0, 1, 4 tokens | 1 token |
| Tier-1 score mode | dependency, fused, none | dependency |

Report as:

| Parameter changed | Best setting | Default setting score | Best score | Sensitivity note |
|---|---:|---:|---:|---|

## 7. Required Figures And Graphs

### Figure 1: Accuracy/F1 vs Compression

One line plot per dataset or a 3-panel figure.

X-axis:

- Compression multiplier: 2x, 4x, 8x, 16x.

Y-axis:

- GSM8K: accuracy.
- NIAH: retrieval accuracy.
- HotpotQA: token F1.

Lines:

- StreamingLLM
- H2O
- SnapKV
- ChunkKV
- TDC-KV
- FullKV horizontal line

### Figure 2: NIAH Position Heatmap

Purpose: match the style of ChunkKV/PyramidKV NIAH evaluation.

Axes:

- X-axis: context length bucket, e.g. 4k, 8k, 16k, 32k if supported.
- Y-axis: needle depth percentage, e.g. 0%, 10%, ..., 100%.
- Color: accuracy.

Make separate heatmaps for:

- TDC-KV
- ChunkKV
- SnapKV
- H2O

### Figure 3: Tier Distribution

Use the existing `plot_tier_distribution` idea.

Show stacked bars:

- Tier 0 eviction candidates.
- Tier 1 soft-protected chunks.
- Tier 2 hard-protected sink/recent chunks.

Report per dataset or per representative sample.

### Figure 4: Evidence Survival / GER vs Compression

Purpose: connect to the Physics paper and strengthen mechanistic analysis.

X-axis:

- Compression ratio: 50%, 75%, 87.5%, 93.75%.

Y-axis:

- GER or answer-token eviction rate.

Lines:

- H2O
- SnapKV
- ChunkKV
- TDC-KV

Expected claim:

- TDC-KV should have lower GER than token-level baselines at the same retention because whole dependency chunks preserve answer routes.

### Figure 5: Latency And VRAM

Two panels:

- End-to-end latency vs generation length.
- Peak VRAM vs generation length.

Use KVpop-style settings:

- Generation lengths: 4k, 8k, 16k, 32k if hardware allows.
- Batch size: 1.
- Retention: 25%.

### Figure 6: Ablation Bar Chart

Y-axis:

- Average normalized quality across GSM8K, NIAH, HotpotQA.

Bars:

- Full TDC-KV.
- No forward routing.
- No tiers.
- Fixed chunks.
- Token-level eviction.
- Uniform layer weights.

## 8. Experiment Pipeline

### Phase A: Correctness And Smoke Tests

Run before any real result claim:

```powershell
cd "C:\Users\jaygo\Desktop\DESKTOP\RMS\Code\Tier-based-KV-cache-with-Dependency-Aware-Chunk-Scoring"
python -m pytest
python -c "import src.core; import src.baselines; print('imports ok')"
python scripts\run_main_results.py --trace-path data\sample_trace.jsonl --recent-window 4 --output outputs\smoke_main.json
python scripts\run_baselines.py --trace-path data\sample_trace.jsonl --recent-window 4 --output outputs\smoke_baselines.json
python scripts\run_ablations.py --trace-path data\sample_trace.jsonl --theta-grid 0.3 --recent-window-grid 4 --output outputs\smoke_ablations.json
```

### Phase B: Trace-Driven Benchmark Runs

Use this after dataset trace files are created:

```powershell
python benchmarks\control\run_gsm8k.py --trace-path data\gsm8k_trace.jsonl --budget 1024 --theta 0.3 --recent-window 16 --output outputs\gsm8k_tdc.json
python benchmarks\control\run_niah.py --trace-path data\niah_trace.jsonl --budget 1024 --theta 0.3 --recent-window 16 --output outputs\niah_tdc.json
python benchmarks\multi_hop\run_hotpotqa.py --trace-path data\hotpotqa_trace.jsonl --budget 1024 --theta 0.3 --recent-window 16 --output outputs\hotpotqa_tdc.json
```

But for the final paper, prefer budget ratios instead of one absolute budget, because datasets have different sequence lengths.

### Phase C: Live HF Matched-Budget Runs

Use `scripts/run_hf_grid.py` as the main experiment entry point after adding baseline generation support.

Suggested starter command:

```powershell
python scripts\run_hf_grid.py `
  --models "meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2.5-7B-Instruct" `
  --datasets "name=gsm8k,source=gsm8k,config=main,split=test,prompt_field=question,answer_field=answer;name=hotpotqa,source=hotpot_qa,config=fullwiki,split=validation,prompt_field=question,answer_field=answer" `
  --budget-ratios "0.5,0.25,0.125,0.0625" `
  --thetas "0.3" `
  --recent-windows "16" `
  --alphas "0.6" `
  --max-chunk-tokens 64 `
  --min-budget-utilization 0.99 `
  --max-budget-shortfall-tokens 1 `
  --allow-level2-fallback `
  --max-samples 200 `
  --max-length 8192 `
  --prefill-block-size 128 `
  --max-new-tokens 64 `
  --device auto `
  --dtype auto `
  --output outputs\hf_grid_main.json
```

The prefill block size controls the attention-memory/forward-call tradeoff and
must be held fixed within each reported comparison. Log both its configured
value and the actual block count; use 64 or 128 for Colab smoke runs, then profile
128 and 256 on the final hardware before fixing the paper configuration.

Every HF output includes `grouped_results`. Use these rows as the source for
tables and plots because they preserve the requested ratio/absolute budget while
also reporting the distribution of per-sample resolved token budgets. Raw `runs`
remain the source for paired significance tests and failure inspection.

NIAH may need a local JSON/JSONL generator rather than a HuggingFace dataset spec.

### Phase D: Plot Generation

Current plotting script:

```powershell
python scripts\plot_results.py outputs\hf_grid_main.json --output-dir outputs\figures
```

Needed additions:

- Multi-method plotting, not only FullKV vs TDC-KV.
- Dataset-specific plot grouping.
- GER/evidence-retention plots.
- NIAH heatmaps.
- VRAM/throughput plots.

## 9. Implementation Gaps Before Final Results

The current codebase is a good prototype, but publication-quality comparison needs these additions:

1. Add StreamingLLM baseline.
   - Keep first `sink_tokens` plus last `recent_window` tokens.
   - Fill remaining capacity either none or highest recency, depending matched baseline definition.

2. Run baselines inside HF generation loop.
   - Current `run_hf_grid.py` produces FullKV and TDC-KV QA.
   - It does not generate answers for H2O, SnapKV, ChunkKV, or StreamingLLM using evicted caches.
   - Add `--methods fullkv,streamingllm,h2o,snapkv,chunkkv,tdc_kv`.

3. Track peak VRAM.
   - Use `torch.cuda.reset_peak_memory_stats()`.
   - Use `torch.cuda.max_memory_allocated()` after prefill/generation.

4. Track throughput.
   - Decode tokens/sec = generated tokens / decode seconds.
   - Report both policy overhead and end-to-end speed.

5. Implement evidence-token metrics.
   - For GSM8K: answer expression tokens are not always in context, so use final-answer EM/accuracy.
   - For NIAH: mark needle fact tokens as answer-critical.
   - For HotpotQA: mark answer string tokens and supporting-fact sentence tokens as answer-critical.

6. Implement GER.
   - For each answer-critical token, check whether it survives in any retained KV route.
   - In current implementation, eviction is global across layers once cache is selected; if using all-layer cache, compute per-layer/head survival when possible.
   - GER = number of globally evicted answer-critical tokens / number of answer-critical tokens.

7. Implement layerwise result logging.
   - Needed for PyramidKV comparison and layer-weight ablation.
   - Log kept counts per layer and attention-mode choice.

8. Enforce exact-budget reporting.
   - TDC-KV uses whole-chunk ranking plus one partial boundary chunk to avoid granularity underfill.
   - Tier 2 fallback is required for matched-budget results; disabling it is a separate protection stress test.
   - Always report `budget_gap`, `budget_utilization`, `budget_shortfall`, and `budget_overflow`.
   - Matched-budget tables require Tier-2 fallback, at least 99% utilization, at most one token shortfall, and zero overflow.

## 10. Ablation Studies In Detail

### Ablation A: Scoring Signal

Question:

- Does forward-routing dependency scoring improve multi-hop preservation?

Run:

| Variant | alpha | beta |
|---|---:|---:|
| Attention only | 1.0 | 0.0 |
| Routing only | 0.0 | 1.0 |
| Balanced | 0.5 | 0.5 |
| Default | 0.6 | 0.4 |
| Routing-heavy | 0.25 | 0.75 |

Expected:

- Attention-only should be competitive on NIAH.
- Full/default should help more on HotpotQA because supporting bridge chunks need dependency preservation.

### Ablation B: Chunk Construction

Question:

- Are sentence-boundary chunks better than fixed token windows?

Run:

| Variant | Chunking rule |
|---|---|
| Sentence-boundary | punctuation tokenizer IDs, current TDC-KV |
| Fixed-8 | every 8 tokens |
| Fixed-16 | every 16 tokens |
| Fixed-32 | every 32 tokens |
| Token-level | chunk size = 1 |

Expected:

- Fixed chunks may work on retrieval but should degrade semantic QA/multi-hop.
- Token-level should have higher fragmentation and higher GER.

### Ablation C: Tier Protection

Question:

- Do Tier 1 and Tier 2 protect against collapse?

Run:

| Variant | Change |
|---|---|
| Full TDC-KV | no change |
| No Tier 1 | `--tier1-score-mode none` (equivalent to `theta=0`) |
| No sink Tier 2 | do not hard-protect sink chunk |
| No recent Tier 2 | recent window = 0 |
| Tier fallback | `allow_level2_fallback=True` |

Expected:

- No recent protection may hurt generation stability.
- No sink protection may hurt all long-context tasks.
- Tier fallback improves exact budget compliance but can increase failure at very low retention.

### Ablation D: Layer Weighting

Question:

- Does PyramidKV-inspired layer awareness help?

Run:

| Variant | Attention source | Layer weights |
|---|---|---|
| Last-layer only | `attention_mode=last` | none |
| All layers uniform | `attention_mode=all` | uniform |
| All layers weighted | `attention_mode=all` | higher-layer weighted |
| Pyramid budget allocation | optional baseline | lower layers receive larger budget |

Expected:

- Last-layer may be enough for speed.
- All-layer weighted may help long-context reasoning but costs more memory/compute.

### Ablation E: Budget Stress

Question:

- Where does TDC-KV fail?

Run retention ratios:

- 50%
- 25%
- 12.5%
- 6.25%
- Optional: 3.125%

Report:

- Task score.
- GER.
- Hallucination rate.
- Budget gap.

Expected:

- TDC-KV should degrade more smoothly than token-only baselines until the extreme budget where hard-protected chunks dominate.

## 11. Parameter Testing Grid

Use two-stage tuning.

Stage 1: small sample grid.

- Samples: 50 per dataset.
- Models: one model, preferably LLaMA-3-8B-Instruct.
- Budget ratios: 0.25 and 0.125.
- Grid:
  - `theta`: 0.1, 0.2, 0.3, 0.4, 0.5
  - `recent_window`: 8, 16, 32, 64
  - `alpha`: 0.25, 0.5, 0.6, 0.75, 1.0
  - `min_chunk_tokens`: 1, 4, 8, 16
  - `max_chunk_tokens`: 32, 64, 128

Stage 2: final confirmation.

- Use best two configs plus default config.
- Run full sample set.
- Run all main models.
- Report if default is within one standard deviation of best; if yes, keep default for simplicity.

## 12. Reproducibility Requirements

For every run, log:

- Git commit hash.
- Model name and revision if available.
- Dataset source, split, and sample count.
- Prompt template.
- Max input length and truncation rule.
- Max new tokens.
- Decoding: greedy, temperature 0, top-p disabled.
- Device and GPU name.
- Torch, transformers, CUDA versions.
- Dtype.
- Budget ratio and absolute resolved budget.
- TDC-KV config: theta, alpha, beta, recent window, min chunk tokens, layer mode.
- Baseline config: method-specific parameters.
- Random seed.

Use three repeats for final tables. Report mean plus standard deviation.

## 13. Recommended Result Section Structure

1. Experimental Setup
   - Models, datasets, baselines, budgets, metrics.

2. Main Results
   - Table 1: three-dataset quality and efficiency at 25% retention.
   - Figure 1: quality vs compression curve.

3. Long-Context Retrieval
   - NIAH heatmaps.
   - Explain positional/depth robustness.

4. Multi-Hop Dependency Preservation
   - HotpotQA results.
   - GER/evidence survival plot.

5. Efficiency
   - Latency, throughput, VRAM.

6. Ablations
   - Scoring signal.
   - Chunking.
   - Tiers.
   - Layer weighting.

7. Failure Analysis
   - Budget-gap cases.
   - Safety cliff at extreme compression.
   - Examples where evidence survived but answer failed.

## 14. Priority Execution Checklist

1. Run unit tests and smoke trace scripts.
2. Add StreamingLLM baseline.
3. Extend HF grid to run all baselines through evicted-cache generation.
4. Add VRAM and throughput logging.
5. Add evidence-token retention and GER logging.
6. Generate or prepare three dataset inputs: GSM8K, NIAH, HotpotQA.
7. Run small grid on 50 samples.
8. Choose final default parameters.
9. Run full main experiments over 3 models x 3 datasets x 4 budgets x 6 methods.
10. Generate result tables.
11. Generate figures.
12. Run ablations.
13. Write failure analysis with example cases.

## 15. What To Claim Carefully

Strong claim:

- TDC-KV is a lightweight training-free method that combines semantic chunks, dependency-aware scoring, and tiered protection to improve quality-memory tradeoffs under matched KV budgets.

Avoid overclaiming:

- Do not claim to beat KVpop unless we run its public implementation or checkpoint under the same models/datasets/budgets.
- Do not claim PyramidKV-style layer allocation unless a true layerwise budget baseline is implemented.
- Do not include a run in matched-budget tables unless its utilization contract passes. Report Tier-2-preserving over-budget runs separately as a protection stress test.

Best comparison framing:

- Compared with ChunkKV, TDC-KV adds dependency-aware forward routing and tier protection.
- Compared with SnapKV/H2O, TDC-KV avoids isolated token pruning by preserving complete chunks.
- Compared with PyramidKV, TDC-KV is not primarily a layer-budget method, but can incorporate layer-weighted scoring.
- Compared with KVpop, TDC-KV is training-free and simpler, while KVpop is a learned future-attention method.
