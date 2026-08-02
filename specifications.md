### Context: The Trigger State
Before Module 1 begins, the transformer layer is doing its normal forward pass. 
Let the current sequence length be **$t$** and the Cache Budget be **$B$**.
The pipeline *only* triggers when a new token causes **$t > B$**.

**Global State Inputs entering the TDC-KV interceptor:**
*   $X \in \mathbb{N}^{1 \times t}$ (The integer Token IDs)
*   $K_{cache}, V_{cache} \in \mathbb{R}^{H \times t \times d}$ (Where $H$ = Heads, $d$ = Head Dimension)
*   $A_{obs} \in \mathbb{R}^{H \times w \times t}$ (The Attention probability weights for the *last $w$ queries only*, bypassing $O(t^2)$ memory).

---

### 🟢 Module 1: Sentence-Boundary Chunk Constructor

*   **Goal:** Convert continuous sequence indices $1 \dots t$ into a set of discrete, variable-length chunks representing complete grammar blocks.

**Inputs:**
*   **$X$:** `tensor` of shape `[t]`, integer Token IDs.

**Parameters:**
*   **$P_{vocab}$:** A constant `set` of integers. These are the token IDs for `{ . , ? , ! , ; , : }` loaded directly from the model's Tokenizer.

**Mathematical Transformation:**
1.  Boolean masking: Create $B \in \{0, 1\}^{t}$ where $B[i] = 1$ if $X[i] \in P_{vocab}$, else $0$.
2.  Find indices where $B=1$. Let this be vector $\mathbf{p} = [p_1, p_2, \dots, p_m]$.
3.  Define bounds: $p_0 = 0$, $p_{m+1} = t$.
4.  Construct list of index arrays: $C_k = [p_{k-1}+1 \dots p_k]$ for $k=1 \dots m+1$.

**Outputs (Data leaving Mod 1):**
*   **$\mathbf{C}$**: A `list of 1D tensors/arrays` representing the token indices belonging to each chunk. (Total $M$ chunks).
*   **$\mathbf{Map}$**: A 1D `tensor` of shape `[t]` where $Map[i] = k$, telling us which chunk token $i$ belongs to.

**Data Transition ➡️:**
$\mathbf{C}$ and $\mathbf{Map}$ are passed immediately to **Module 2** to tell it how to aggregate the floating-point scores.

---

### 🟡 Module 2: Dual Signal Scorer (The Core Math Engine)

*   **Goal:** Read raw attention, calculate individual Token Scores for $t$, then squish those into Chunk Scores using $\mathbf{C}$ and $\mathbf{Map}$.

**Inputs:**
*   **$A_{obs}$**: `float16/bfloat16 tensor` of shape `[H, w, t]` (Attention map of observation window).
*   **$\mathbf{C}$**: List of length $M$ from Module 1.
*   **$G$**: Sparse chunk-dependency graph with at most $k$ outgoing edges per chunk.

**Parameters:**
*   **$\alpha = 0.6$**, **$\beta = 0.4$** (Mixing coefficients).
*   **$w = 16$** (Observation window length).
*   **$\mathbf{L}_{weights}$**: `1D float tensor` across all layers where higher layers $> 1$ and lower layers $< 1$ (from PyramidKV insight).

**Mathematical Transformation:**
*   **Signal 1 (Attention Mass):** How much do the $w$ recent queries look at historical token $j$? 
    $$M \in \mathbb{R}^{t} \quad \text{where} \quad M[j] = \sum_{h=1}^H \sum_{q=t-w+1}^{t} A_{obs}[h, q, j]$$
    *   *Operation:* Sum $A_{obs}$ over the $w$ and $H$ dimensions. Output shape `[t]`.
*   **Sparse Dependency Collection:** During prefill, aggregate each query chunk's attention into key-chunk mass. Remove the self-edge, retain the strongest $k$ edges, and normalize each sparse row:
    $$G[q,r] = \operatorname{mean}_{j \in C_q, h}\sum_{i \in C_r} A[h,j,i]$$
    $$\sum_{r \in N(q)} G[q,r] = 1, \quad |N(q)| \le k$$
*   **Signal 2 (Historical Dependency Routing):** First aggregate direct attention mass into chunks, then route current relevance through historical dependencies:
    $$S_1[q] = \text{mean}(M[C_q])$$
    $$S_2[q] = \sum_{r \in N(q)} G[q,r] \times \hat{S}_1[r]$$
    Chunks without historical edges fall back to their direct relevance $\hat{S}_1[q]$.
*   **Fusion:** Min-max normalize both to $[0,1]$, resulting in $\hat{S}_1$ and $\hat{S}_2$.
    $$\mathbf{Score_{chunk}}[k] = (\alpha \times \hat{S}_1[k]) + (\beta \times \hat{S}_2[k])$$

**Outputs (Data leaving Mod 2):**
*   **$\mathbf{Score_{chunk}}$:** `float tensor` of shape `[M]` representing mathematical importance per chunk.
*   **Graph storage:** $O(Mk)$ rather than a persistent dense $O(M^2)$ chunk graph.

**Current implementation note:** HuggingFace prefill is causal and blockwise. For each query block, selected attention layers are aggregated and immediately streamed into the sparse graph builder; only rows overlapping the final observation window are retained. Dense attention memory is therefore bounded by $O(LHBt)$ for block size $B$, rather than retaining the full $O(LHt^2)$ attention tensor. The cumulative full prompt KV cache remains resident until eviction.

**Data Transition ➡️:**
The continuous values of $\mathbf{Score_{chunk}}$ are passed to **Module 3** to be quantized into 3 physical safety buckets.

---

### 🟠 Module 3: Protection Mask Assigner

*   **Goal:** Convert continuous decimals ($0.423, 0.991, 0.121$) into discrete logical Tiers $(0, 1, 2)$.

**Inputs:**
*   **$\mathbf{Score_{chunk}}$**: Fused score of shape `[M]`, used for within-tier eviction ordering.
*   **$\hat{S}_2$**: Dependency-routing score of shape `[M]`, used to assign Tier 1.
*   **$\mathbf{C}$**: Structure from Mod 1 identifying token positions per chunk.

**Parameters:**
*   **$\theta$ = 0.3**: (The "Top 30%" threshold ratio for soft-protection).
*   **$w$ = 16**: (The exact number of recent tokens hard-protected by StreamingLLM logic).

**Mathematical Transformation:**
1.  Initialize **$\mathbf{\Pi}$**: An integer tensor of shape `[M]` initialized entirely to $0$.
2.  **Level 1 Masking:** Exclude Tier-2 sink/recent chunks, then select exactly $\lceil\theta M_{eligible}\rceil$ eligible chunks with the highest dependency-routing scores.
    $$ \mathbf{\Pi}[\operatorname{TopK}(\hat{S}_2, \lceil\theta M_{eligible}\rceil)] = 1 $$
    Tier 1 is therefore independent of fused-score ordering: a bridge chunk with modest direct attention can be soft-protected because of its historical dependency route.
3.  **Level 2 Masking:** Override based on static positions. Let chunk 0 containing token $0$ (the attention sink) be $C_{sink}$ and the chunk containing the start of the observation window ($t-w$) be $C_{recent}$.
    $$\mathbf{\Pi}[C_{sink}] = 2, \quad \mathbf{\Pi}[C_{recent} \dots M] = 2$$

**Outputs (Data leaving Mod 3):**
*   **$\mathbf{\Pi}$ (The Mask)**: `int8 tensor` of shape `[M]`, values strictly $\{0, 1, 2\}$.

**Data Transition ➡️:**
Everything is now shipped to the Executor (**Module 4**), containing the final verdict values: $\mathbf{C}$ (where everything is), $\mathbf{\Pi}$ (tier lists), $\mathbf{Score}$ (fine-grained priority).

---

### 🔴 Module 4: Priority-Respecting Eviction Engine

*   **Goal:** Select precise matrix indices to drop and output physically smaller matrices.

**Inputs:**
*   **$\mathbf{\Pi}$**: Shape `[M]` (From Mod 3)
*   **$\mathbf{Score_{chunk}}$**: Shape `[M]` (From Mod 2)
*   **$\mathbf{C}$**: Length $M$ (From Mod 1)
*   **$K_{cache}, V_{cache}$**: `[H, t, d]` (The full-size hardware tensors needing truncation).

**Parameters:**
*   **$B$**: (Maximum Context Cache Budget). Let Deficit = $(t - B)$. We need to delete exactly *Deficit* number of tokens.

**Mathematical Transformation:**
1.  Create an empty boolean mask of shape `[t]` called `KeepMask`, set all values to `True`. Let TokensRemoved = 0.
2.  Identify **Target 0 chunks**: $Q_0 = \{ k \mid \mathbf{\Pi}[k] == 0 \}$. Sort $Q_0$ in *ascending* order based on $\mathbf{Score_{chunk}}[k]$.
3.  Loop through sorted $Q_0$:
    *   Let $idx$ = token indices for chunk $C_{curr}$.
    *   Set `KeepMask`$[idx]$ = `False`.
    *   TokensRemoved += $|C_{curr}|$
    *   If TokensRemoved $\ge$ Deficit: `BREAK loop`
4.  *Safety Net:* If TokensRemoved < Deficit (we ran out of Level 0s), repeat Steps 2 & 3 but targeting Level 1 chunks ($Q_1$). *(Level 2s are literally skipped, loop impossible here).*
5.  Get non-deleted integer indices: `idx_retained = torch.nonzero(KeepMask)`. This creates a flattened index of shape `[Final_Count]` where $Final\_Count \approx B$.

**Outputs (Returning to the base LLM Model):**
*   **$K_{cache}^{new} = K_{cache}[\text{:} \:, idx\_retained, \text{: }]$** (New Shape: `[H, Final_Count, d]`).
*   **$V_{cache}^{new} = V_{cache}[\text{:} \:, idx\_retained, \text{: }]$**

### Decode-Time Budget Enforcement

After prefill eviction, the decoding cache manager stores one metadata entry per physical KV position: original logical position, chunk/group id, fused score, and persistent Tier-1 status. Sink and recent Tier-2 status are recomputed from logical positions after every generated token.

For each decode step:

1. Process one generated token, temporarily growing the cache by one position.
2. Append matching metadata for that position.
3. If the cache exceeds $B$, remove complete Tier-0 groups by ascending fused score, then Tier-1 groups.
4. If Tier-2 tokens alone exceed $B$, remove the oldest non-sink Tier-2 groups and record a fallback event.
5. Apply the retained physical indices to every key/value layer and rebuild the model cache.

The enforced invariant is:

$$|K_{cache}^{post-step}| = |V_{cache}^{post-step}| \le B$$

Generated tokens use individual groups in the current implementation. They remain Tier 2 while inside the recent window and become low-priority eviction candidates after aging out.

Offline integration tests exercise this invariant with tiny GPT-2, Llama, and
Qwen2 causal language models. They compare compressed-cache logits against an
independent forward pass, verify logical `cache_position` values for RoPE
families, and require exact budget occupancy after repeated token-granular
decode eviction. Whole-chunk prefill eviction may still underfill the budget.

All cache tensor access passes through a shared HuggingFace compatibility
adapter. It supports legacy tuple caches, v4 cache lists, and v5
`DynamicCache.layers` while exposing the same ordered key/value layer contract
to prefill extraction and decode-time compaction.

### Final Check

Does the data flow perfectly connect?
**YES:** 
Tokens ($[t]$) $\rightarrow$ Chunk Definitions $\rightarrow$ Mathematical Aggregation ($\mathbf{Score}$) $\rightarrow$ Rule-based Constraints ($\mathbf{\Pi}$) $\rightarrow$ Sorted Elimination $\rightarrow$ Target Memory State (`[B]`).

By implementing in this distinct step-wise sequence inside PyTorch, every computation scales purely as 1-Dimensional $O(t)$ tensor math on the GPU (minus the restricted window of $A_{obs}$). This is what keeps it blindingly fast for your Kaggle T4 budget limits.
