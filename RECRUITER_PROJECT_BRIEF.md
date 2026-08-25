# TDC-KV: Tier-Based KV-Cache Compression with Dependency-Aware Chunk Scoring

## Recruiter-ready summary

**TDC-KV is a training-free systems and ML-infrastructure project that makes long-context LLM inference more memory-efficient.** It selectively compresses an LLM's key-value (KV) cache, the memory structure that grows linearly with every token an LLM reads or generates. Rather than deleting isolated tokens, TDC-KV identifies meaningful text chunks, estimates which earlier chunks still support the current reasoning, and protects the most important material under a strict memory budget.

The project addresses a practical bottleneck in deploying long-context assistants: models can often *accept* long prompts, but the KV cache consumes GPU memory throughout generation. TDC-KV is designed to retain useful context while reducing that memory footprint by **2x, 4x, 8x, or 16x** at the evaluated retention points (50%, 25%, 12.5%, and 6.25%, respectively).

> **Important evidence note:** the repository contains a strong implementation and qualification pipeline, plus exploratory small-model pilots. It does **not** yet claim paper-grade quality improvements over official baselines or completed 7B/8B headline results. This distinction is intentional and is a strength of the project’s engineering discipline.

## The problem in plain language

When an autoregressive transformer processes a prompt, it stores a **key** and **value** representation for every past token at every layer. Future tokens attend to that stored cache instead of recomputing the entire prompt. This makes generation possible, but creates a growing memory cost:

- Longer prompt -> more KV-cache memory.
- More generated tokens -> more KV-cache memory.
- Larger model -> more layers and/or KV heads -> more KV-cache memory per token.

Naively dropping old tokens can break a response because an older sentence may contain a definition, a constraint, a retrieved fact, or one hop in a multi-step reasoning chain. Keeping every token is safe but expensive; deleting by recency alone is cheap but can discard critical evidence.

**TDC-KV treats cache compression as a constrained importance-selection problem:** keep exactly the allowed number of cache positions while prioritizing text that is recent, structurally important, and still connected to currently relevant reasoning.

## Real-world use cases

TDC-KV is relevant wherever LLMs need to reason over long conversations or documents on finite GPU memory:

- **Enterprise assistants:** retain important policies, prior decisions, and facts across a long support or analyst session.
- **Retrieval-augmented generation (RAG):** preserve passages that support an answer while shrinking a large retrieved context.
- **Document intelligence:** process contracts, research papers, manuals, medical records, or financial filings without treating all early content as disposable.
- **Agentic workflows:** retain earlier tool outputs, plans, and dependencies during multi-step tasks.
- **Cost-sensitive or edge inference:** fit longer usable contexts on smaller GPUs, reducing the KV-memory portion of hardware pressure.

## What makes TDC-KV different

Most simple cache policies are token-centric: they may keep the first “sink” tokens, the most recent window, or tokens with high local attention. TDC-KV adds two ideas:

1. **Semantic chunks instead of isolated tokens.** The sequence is divided at sentence/punctuation boundaries (with a maximum chunk length). This makes eviction more coherent: a sentence or clause is more likely to survive as a usable unit.
2. **Dependency-aware importance.** If a currently relevant chunk depended on an earlier chunk, TDC-KV routes some importance backward to that earlier “bridge” chunk. This is intended to preserve multi-hop chains, not only text that receives strong immediate attention.
3. **Tiered protection.** Sink and recent context are hard-protected; high dependency-score chunks receive soft protection; low-priority chunks are evicted first.
4. **Exact token budget enforcement.** It ranks whole chunks but trims at most one boundary chunk when necessary, so the retained cache can match a token-level budget rather than merely approximating it.

## How the system works

```text
Prompt tokens + live attention
          |
          v
  1. Build bounded semantic chunks
          |
          v
  2. Score direct attention and historical dependencies
          |
          v
  3. Assign Tier 0 / Tier 1 / Tier 2 protection
          |
          v
  4. Evict lowest-priority cache positions to the exact budget
          |
          v
  Compressed KV cache used for continued generation
```

### 1. Chunk construction

The chunker breaks the token stream at punctuation boundaries (`.`, `,`, `?`, `!`, `;`, `:`) and splits punctuation-free spans that exceed a configured maximum (default: 64 tokens). Every token belongs to one contiguous chunk.

This avoids the failure mode of deleting a few tokens from the middle of a fact or sentence and leaving an incoherent fragment behind.

### 2. Dual-signal scoring

For each token position `j`, direct attention mass is calculated from the last `w` query positions across attention heads:

```text
M[j] = sum over heads h and recent queries q of A[h, q, j]
```

The system aggregates these token scores into chunk scores. During prefill, it also builds a sparse chunk-dependency graph. If a later query chunk `q` attends to an earlier key chunk `r`, an edge `q -> r` records that relationship. Each chunk retains only its strongest `k` historical edges, making graph storage proportional to `O(Mk)` rather than `O(M^2)` for `M` chunks.

Dependency routing boosts an earlier chunk when relevant later chunks depend on it:

```text
S2[r] = normalized direct relevance of r
        + sum of (dependency strength q -> r * relevance of q)
```

The final chunk score mixes direct and routed relevance:

```text
score(chunk) = alpha * normalized direct score
             + beta  * normalized dependency-routed score
```

The default documented configuration uses `alpha = 0.6`, `beta = 0.4`, a 16-token observation window, and a 30% Tier-1 selection threshold. These are tunable and the experiment plan includes ablations for chunking, tiering, signals, and layer weighting.

### 3. Three protection tiers

| Tier | Meaning | Eviction behavior |
|---|---|---|
| Tier 2 | Attention-sink and recent-context chunks | Hard-protected unless an explicit matched-budget fallback is allowed |
| Tier 1 | Chunks with highest dependency-routing scores | Soft-protected; evicted only after Tier 0 if needed |
| Tier 0 | Remaining lower-priority chunks | Primary eviction candidates, ordered from lowest score upward |

### 4. Strict-budget eviction and decode-time control

If the sequence has `t` cache positions and the budget is `B`, TDC-KV removes exactly `t - B` positions (for matched-budget runs). It first evicts Tier 0 chunks, then Tier 1 only if necessary. It evicts complete low-value chunks until the final boundary, where it can remove only the oldest necessary positions from one chunk to hit the exact target.

The cache manager repeats this control during decoding, not only after the prompt prefill. That matters because the cache continues to grow as the model generates.

## Projected memory impact: the math

The KV cache has two tensors (key and value). For a model with:

- `L` transformer layers
- `H_kv` KV heads
- `d` values per head
- `T` cached tokens
- `b` bytes per stored value (2 for FP16/BF16)

the approximate KV-cache size is:

```text
KV bytes = 2 * L * H_kv * d * T * b
```

The factor of 2 represents key and value tensors. TDC-KV changes `T` to `rT`, where `r` is the retention ratio. Therefore, KV memory scales linearly:

```text
compressed KV bytes = r * full KV bytes
memory saved          = (1 - r) * full KV bytes
compression multiplier = 1 / r
```

### Budget-to-impact table

| KV retention | KV memory saved | Compression multiplier | Meaning |
|---:|---:|---:|---|
| 50% | 50% | 2x | Keep half the cache; halve KV-memory use |
| 25% | 75% | 4x | Keep one quarter; use one quarter of the KV memory |
| 12.5% | 87.5% | 8x | Keep one eighth of the cache |
| 6.25% | 93.75% | 16x | Keep one sixteenth of the cache |

These are **mathematical projections for the KV-cache component**, not claims that total GPU memory or end-to-end latency improve by the same percentage. Model weights, activations, attention computation, framework overhead, and batch size still matter.

### Concrete illustrative example

For a typical grouped-query 8B-class configuration with 32 layers, 8 KV heads, head dimension 128, and FP16 values:

```text
KV bytes/token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KiB/token
```

At an 8,192-token context this is approximately **1 GiB of KV cache per sequence**. Under the same architecture and precision:

| Retention | Approximate KV cache at 8K tokens | KV memory released |
|---:|---:|---:|
| 100% (FullKV) | 1.00 GiB | 0 GiB |
| 50% | 0.50 GiB | 0.50 GiB |
| 25% | 0.25 GiB | 0.75 GiB |
| 12.5% | 0.125 GiB | 0.875 GiB |
| 6.25% | 0.0625 GiB | 0.9375 GiB |

This example is an architectural calculation, not a benchmark measurement. Exact values vary by model configuration, precision, framework cache layout, context length, and batch size.

## Engineering depth and implementation scope

This is not a conceptual notebook only. The repository implements and tests the full cache-management path:

- Core modules for chunking, sparse dependency collection, scoring, tier assignment, and eviction.
- A cache manager that re-enforces the budget during token generation.
- Hugging Face cache adapters and compatibility coverage for Transformers v4/v5 cache forms.
- Offline end-to-end compressed-cache tests using tiny GPT-2, Llama, and Qwen2 fixtures.
- Blockwise prefill: attention blocks are consumed for scoring and then released, limiting retained dense attention memory to the prefill block rather than the whole prompt-square tensor.
- Baseline implementations for StreamingLLM, H2O, SnapKV, and ChunkKV, clearly labeled as local approximations rather than official reproductions.
- Dataset-aware evaluation for GSM8K, Needle-in-a-Haystack (NIAH), and HotpotQA.
- Prompt hashing, immutable model-revision handling, frozen dataset manifests, deterministic run keys, atomic checkpoints, resume support, runtime/VRAM metrics, and qualification gates.

The reported test contract in the current experiment runbook is **139 passing tests**. The suite covers core algorithms, budget behavior, reproducibility controls, Hugging Face adapters, and end-to-end cache behavior; passing tests validate implementation correctness for those paths, not model-quality superiority.

## Evaluation design

The project is designed to evaluate both quality and systems behavior under matched budgets.

| Task | What it tests | Primary metric |
|---|---|---|
| GSM8K | Mathematical reasoning with a fixed eight-shot prompt protocol | Extracted numeric accuracy |
| NIAH | Long-context fact/needle retrieval at controlled depth | Exact normalized needle retrieval accuracy |
| HotpotQA | Multi-hop question answering and evidence preservation | Normalized answer F1 |

The planned headline protocol uses three 7B/8B model families, disjoint qualification/tuning/final data partitions, retention budgets from 50% to 6.25%, deterministic greedy decoding, parity checks against an unpruned custom cache, and separate timing, quality, and ablation phases. The artifact pipeline is designed to emit paired statistical tests, bootstrap confidence intervals, memory/throughput tables, evidence-retention diagnostics, and ablation plots.

## What has been observed so far

An August 2026 exploratory pilot ran **80/80 configurations successfully** on `Qwen/Qwen2.5-0.5B-Instruct`: 40 GSM8K runs, 20 NIAH runs, and 20 HotpotQA runs. These results are retained as engineering evidence, not headline research claims.

### Verified pilot signal: exact budget control on HotpotQA

| Configuration | Actual retention | Compression | Official F1 | F1 relative to FullKV | Eviction policy time |
|---|---:|---:|---:|---:|---:|
| FullKV | 100% | 1.00x | 0.0627 | 100.0% | 0.000 ms |
| TDC-KV at 25% | 25% | 4.00x | 0.0345 | 55.1% | 7.575 ms |
| TDC-KV at 50% | 50% | 2.00x | 0.0348 | 55.5% | 6.012 ms |
| TDC-KV at 75% | 75% | 1.33x | 0.0487 | 77.6% | 5.180 ms |

This pilot demonstrates that the eviction mechanism can honor exact retention budgets in a live QA run. It does **not** establish a quality advantage: absolute scores were low, the sample was only five examples, and the model was intentionally small.

### Findings that should not be hidden

- The older GSM8K pilot had weak quality on the 0.5B model; one compressed configuration answered 1 of 10 examples correctly, while FullKV answered none. That isolated result is not evidence of improvement.
- The older NIAH pilot had **0/15 compressed retrieval successes** and severe budget underfill. FullKV retrieved 3/5 needles, but two FullKV failures were consistent with prompt truncation at the configured limit.
- Those failures motivated the current qualification work: exact-budget checks, no-truncation rejection, tokenizer-exact NIAH construction, evidence-token logging, prompt identity hashes, and parity controls.

This transparent treatment is appropriate for a recruiter: it shows the ability to recognize negative results, diagnose their causes, and build the instrumentation required to make the next experiment trustworthy.

## Reproducibility and measurement discipline

The project goes beyond a one-off benchmark script. Its qualification framework rejects or flags runs that are not comparable:

- prompt serialization is performed once and hashed at raw text, rendered prompt, and token-ID levels;
- model revisions and dataset record manifests are pinned and hash-verified;
- qualification, tuning, and final partitions are separated to avoid tuning on final results;
- custom-cache generation is checked against FullKV parity on the same token IDs;
- compressed rows must satisfy zero budget overflow, at least 99% budget utilization, and at most one token of shortfall;
- CUDA-synchronized stage timings, peak VRAM, physical KV bytes before/after eviction, and decode throughput are recorded;
- checkpoints are atomic and resume validation prevents incompatible experiments from being silently mixed.

## Technical trade-offs and limitations

TDC-KV deliberately makes several trade-offs:

- It needs attention information to score importance. The current paper path uses eager attention and blockwise prefill, which is more measurement-friendly but can be less memory-efficient than flash-attention-style paths.
- Chunk-level preservation improves coherence but may retain a few lower-value neighboring tokens. The boundary trim restores exact token budgets when required.
- Tier-2 protection preserves recency and attention-sink behavior, but extremely small budgets may require a documented fallback to meet a strict budget.
- The sparse graph is memory-efficient (`O(Mk)`), but it is an approximation of the full history of attention relationships.
- Current local baseline implementations are useful comparison controls but are expressly not presented as bit-for-bit official baseline reproductions.

## Why this matters as a portfolio project

TDC-KV demonstrates applied capability across several recruiter-relevant areas:

- **LLM systems:** transformer KV-cache internals, inference memory constraints, and generation-time cache control.
- **ML engineering:** Hugging Face integration, model-version compatibility, deterministic experiments, and dataset-specific evaluation.
- **Algorithms:** sparse graphs, relevance propagation, constrained ranking/selection, and complexity-aware design.
- **Research rigor:** ablation design, matched-budget comparisons, statistical-artifact generation, negative-result analysis, and careful claim boundaries.
- **Production thinking:** explicit failure gates, provenance, checkpoint/resume behavior, memory and throughput instrumentation, and reproducible experiment contracts.

## Suggested recruiter talking points

> I built TDC-KV, a training-free long-context inference system that reduces the KV-cache memory used by transformer LLMs while trying to preserve the context needed for reasoning. It scores sentence-level chunks using both recent attention and a sparse dependency graph, then applies tiered protection and exact-budget eviction during generation. At a 25% retention budget, the KV-cache portion is mathematically 4x smaller (75% less KV memory); at 6.25%, it is 16x smaller. I also built the Hugging Face cache integration, compatibility tests, measurement pipeline, reproducibility controls, and qualification gates needed to evaluate the method honestly. The initial 0.5B pilot validated the system path but surfaced NIAH retrieval failures, so I strengthened budget, parity, and evidence-retention instrumentation before treating any large-model result as a claim.

## Repository map

| Area | Key locations |
|---|---|
| Algorithm core | `src/core/chunker.py`, `dependency_graph.py`, `scorer.py`, `masker.py`, `evictor.py` |
| Runtime cache handling | `src/models/cache_manager.py`, `hf_cache_adapter.py`, `cache_utils.py` |
| Baselines | `src/baselines/` |
| Experiment system | `benchmarks/`, `scripts/run_hf_grid.py`, `scripts/run_paper_suite.py` |
| Test suite | `tests/` |
| Full protocol | `EXPERIMENT_RUNBOOK.md`, `tdc_kv_results_experiment_plan.md` |
| Mathematical design | `specifications.md` |

## Bottom line

TDC-KV is a technically ambitious, implementation-backed approach to one of the most important practical constraints in long-context LLM inference: KV-cache growth. Its immediate, mathematically guaranteed benefit is linear reduction in the **KV-cache component** at a chosen retention budget. Its research hypothesis is that dependency-aware chunk selection retains more useful context than simpler pruning at the same budget. That hypothesis is supported by a robust implementation and evaluation plan, but still requires qualified large-model benchmarking before it should be presented as a proven quality win.
