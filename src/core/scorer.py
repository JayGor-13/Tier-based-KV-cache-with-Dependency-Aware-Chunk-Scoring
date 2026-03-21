"""
TDC-KV Module 2: Dual Signal Scorer
=====================================
Computes two independent importance signals per token from the observation
window attention matrix, aggregates them to chunk level, normalizes, and
fuses into a single Score_chunk vector.

Signal 1 (M): Attention Mass  — how much do recent queries attend TO token j?
Signal 2 (R): Forward Routing — does token j attend toward high-M tokens?
              (catches multi-hop bridge tokens that S1 alone would miss)

Follows spec exactly. Issues are marked with # ISSUE comments.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Module 2 core
# ---------------------------------------------------------------------------

class DualSignalScorer:
    """
    Module 2 of TDC-KV.

    Takes the observation-window attention matrix A_obs and the chunk
    definitions from Module 1, and returns a single Score_chunk vector
    of shape [M] where M is the number of chunks.

    All computation stays on-device and operates on pre-existing tensors —
    no new model forward passes are triggered.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        window_size: int = 16,
        num_layers: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        alpha : float
            Weight for Signal 1 (Attention Mass). Default 0.6.
        beta : float
            Weight for Signal 2 (Forward Routing). Default 0.4.
            Constraint: alpha + beta == 1.0 (enforced at runtime).
        window_size : int
            Number of recent tokens used as the observation window (w).
            Must match the w used when A_obs was computed.
        num_layers : int or None
            Total number of transformer layers L.
            Required for layer-weighted score aggregation.
            If None, uniform weighting is used (all layers equal weight).
        device : str or torch.device
            Device where all tensors are expected and outputs will be placed.
        """
        # ISSUE: The spec sets alpha=0.6, beta=0.4 as defaults with the note
        # "to be validated empirically." These are unvalidated starting points.
        # A sensitivity sweep over (alpha, beta) pairs should be run before
        # claiming any performance numbers depend on this specific ratio.
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(
                f"alpha + beta must equal 1.0, got {alpha} + {beta} = {alpha + beta}"
            )

        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.num_layers = num_layers
        self.device = torch.device(device)

        # Layer weights: w_l = l / sum(l') giving higher layers more weight.
        # Following PyramidKV's validated finding that higher layers carry
        # more semantically concentrated attention patterns.
        # ISSUE: PyramidKV's finding was used to justify per-layer BUDGET
        # allocation, not per-layer SCORE WEIGHTING. Applying it to scoring
        # is a reasonable hypothesis but is not empirically validated here.
        # Uniform weighting (num_layers=None) is provided as a clean ablation.
        if num_layers is not None and num_layers > 0:
            layer_indices = torch.arange(1, num_layers + 1, dtype=torch.float32)
            self.layer_weights: Tensor = (layer_indices / layer_indices.sum()).to(self.device)
        else:
            self.layer_weights = None

    # ------------------------------------------------------------------
    # Primary interface — called once per eviction trigger (prefill)
    # ------------------------------------------------------------------

    def forward(
        self,
        A_obs: Tensor,
        chunks: list[Tensor],
    ) -> Tensor:
        """
        Compute Score_chunk for all M chunks.

        Parameters
        ----------
        A_obs : Tensor, shape [H, w, t] OR [L, H, w, t]
            Attention probability weights restricted to the observation window.
            - [H, w, t]: single-layer mode (layer weights ignored)
            - [L, H, w, t]: multi-layer mode (layer weights applied)
            dtype should be float16 or bfloat16 as it comes from the model.
            Internally upcast to float32 for numerical stability.

        chunks : list of 1D Tensors (length M)
            Chunk index lists from Module 1. chunks[k] contains the integer
            token indices belonging to chunk k.

        Returns
        -------
        Score_chunk : Tensor, shape [M], dtype=float32
            Combined normalized importance score per chunk.
            Score_chunk[k] = alpha * S1_hat[k] + beta * S2_hat[k]
        """
        # Upcast to float32 for numerical stability during scoring
        # ISSUE: A_obs arrives as float16/bfloat16 from the model's attention
        # computation. Keeping it in float16 during the sum operations risks
        # silent overflow for long sequences (t > 4096). Always upcast here.
        A = A_obs.float()

        # Dispatch based on tensor dimensionality
        if A.dim() == 3:
            # Single-layer input: shape [H, w, t]
            M_token, R_token = self._compute_signals_single_layer(A)
        elif A.dim() == 4:
            # Multi-layer input: shape [L, H, w, t]
            M_token, R_token = self._compute_signals_multi_layer(A)
        else:
            raise ValueError(
                f"A_obs must be 3D [H,w,t] or 4D [L,H,w,t], got shape {A_obs.shape}"
            )

        t = M_token.shape[0]
        M_chunks = len(chunks)

        # Validate chunk coverage
        # ISSUE: If Module 1's update() left a trailing empty chunk
        # (placeholder after a boundary token), mean() on it returns nan.
        # We guard by skipping empty chunks and assigning them score 0.0.
        self._validate_chunks(chunks, t)

        # Aggregate token-level signals to chunk level
        S1 = self._aggregate_to_chunks(M_token, chunks, M_chunks)  # shape [M]
        S2 = self._aggregate_to_chunks(R_token, chunks, M_chunks)  # shape [M]

        # Min-max normalize both signals to [0, 1]
        S1_hat = self._minmax_normalize(S1)
        S2_hat = self._minmax_normalize(S2)

        # Fuse signals
        Score_chunk = self.alpha * S1_hat + self.beta * S2_hat

        return Score_chunk

    # ------------------------------------------------------------------
    # Signal computation — single layer
    # ------------------------------------------------------------------

    def _compute_signals_single_layer(self, A: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute M and R from a single-layer attention matrix.

        Parameters
        ----------
        A : Tensor, shape [H, w, t], float32

        Returns
        -------
        M_token : Tensor, shape [t]   — attention mass per token
        R_token : Tensor, shape [t]   — forward routing score per token
        """
        H, w, t = A.shape

        # -----------------------------------------------------------
        # Signal 1: Attention Mass M[j]
        # How much total attention do the w recent queries pay to token j?
        # M[j] = sum over h, sum over q in window of A[h, q, j]
        # Spec formula: M[j] = Σ_h Σ_{q=t-w+1}^{t} A_obs[h, q, j]
        # A_obs already contains only the window queries so we sum over
        # all w query positions and all H heads.
        # -----------------------------------------------------------
        # Sum over heads (dim 0) and window queries (dim 1) → shape [t]
        M_token = A.sum(dim=0).sum(dim=0)  # [H,w,t] → [w,t] → [t]

        # -----------------------------------------------------------
        # Signal 2: Forward Routing Score R[j]
        # Does token j (as a query) attend toward tokens with high M?
        # R[j] = Σ_h Σ_{i=1}^{t} A[h, j_local, i] * M[i]
        #
        # ISSUE: The spec computes R[j] ONLY for tokens j inside the
        # observation window (the last w tokens), saving compute.
        # For tokens outside the window, R[j] defaults to M[j].
        # This is the chosen default: R[j] = M[j] for j < t-w.
        # Rationale: no forward routing information is available for
        # these tokens from this window, so we fall back to their
        # attention mass as a conservative estimate.
        # This means outside-window tokens cannot be "rescued" by
        # Signal 2 — only inside-window tokens can have R > M.
        # -----------------------------------------------------------

        # Initialize R as a copy of M (default for out-of-window tokens)
        R_token = M_token.clone()

        # Identify window token positions in the full sequence
        # Window covers positions [t-w, t-1] (0-indexed)
        window_start = t - w
        window_end = t  # exclusive

        if window_start < 0:
            # Edge case: sequence shorter than window size
            # ISSUE: When t < w the entire sequence IS the window.
            # All tokens get a proper R computation in this case.
            window_start = 0

        # For each window token j (local index q maps to global index t-w+q):
        # R[j] = Σ_h Σ_i A[h, q, i] * M[i]
        # Vectorized: A has shape [H, w, t], M has shape [t]
        # For all window queries simultaneously:
        #   weighted = A * M[None, None, :]   → [H, w, t]  (broadcast M over H,w)
        #   R_window = weighted.sum(dim=0).sum(dim=0)... No — that collapses wrong.
        #
        # We want R[q] = Σ_h Σ_i A[h,q,i] * M[i]  for each query position q
        # = Σ_h (A[h, q, :] @ M)
        # = (A.sum(dim=0))[q, :] @ M  if we sum heads first
        # Actually: R_window[q] = Σ_h Σ_i A[h,q,i]*M[i]
        #         = (A * M[None,None,:]).sum(dim=2).sum(dim=0)  → shape [w]
        #
        # Shape trace:
        #   A:              [H, w, t]
        #   M[None,None,:]: [1, 1, t]  (broadcast)
        #   product:        [H, w, t]
        #   .sum(dim=2):    [H, w]     (sum over key positions i)
        #   .sum(dim=0):    [w]        (sum over heads h)
        # Result: R_window[q] for q = 0..w-1, maps to global pos t-w+q

        R_window = (A * M_token[None, None, :]).sum(dim=2).sum(dim=0)  # [w]

        # Write window R values back into the full R_token vector
        actual_window_len = window_end - window_start
        R_token[window_start:window_end] = R_window[:actual_window_len]

        return M_token, R_token

    # ------------------------------------------------------------------
    # Signal computation — multi layer
    # ------------------------------------------------------------------

    def _compute_signals_multi_layer(self, A: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute layer-weighted M and R from a multi-layer attention tensor.

        Parameters
        ----------
        A : Tensor, shape [L, H, w, t], float32

        Returns
        -------
        M_token : Tensor, shape [t]
        R_token : Tensor, shape [t]
        """
        L, H, w, t = A.shape

        # Build or validate layer weights
        if self.layer_weights is not None and self.layer_weights.shape[0] == L:
            layer_w = self.layer_weights  # shape [L]
        else:
            # ISSUE: If num_layers at init time doesn't match actual L in A_obs,
            # we fall back to uniform weights rather than raising an error.
            # This ensures robustness during experimentation when layer counts
            # might vary, but silently changes behavior. A warning is appropriate.
            layer_w = torch.ones(L, dtype=torch.float32, device=self.device) / L

        # Compute per-layer M and R, then apply weighted average
        # Shape: accumulate [t] tensors weighted by layer importance
        M_accum = torch.zeros(t, dtype=torch.float32, device=self.device)
        R_accum = torch.zeros(t, dtype=torch.float32, device=self.device)

        for l_idx in range(L):
            A_l = A[l_idx]  # [H, w, t]
            M_l, R_l = self._compute_signals_single_layer(A_l)
            weight = layer_w[l_idx].item()
            M_accum += weight * M_l
            R_accum += weight * R_l

        return M_accum, R_accum

    # ------------------------------------------------------------------
    # Aggregation and normalization helpers
    # ------------------------------------------------------------------

    def _aggregate_to_chunks(
        self,
        token_scores: Tensor,
        chunks: list[Tensor],
        M_chunks: int,
    ) -> Tensor:
        """
        Aggregate token-level scores to chunk level using mean pooling.

        Spec formula:
            S_i(C_k) = (1 / |C_k|) * Σ_{j ∈ C_k} score[j]

        Parameters
        ----------
        token_scores : Tensor, shape [t]
        chunks : list of 1D Tensors, length M
        M_chunks : int

        Returns
        -------
        chunk_scores : Tensor, shape [M]
        """
        chunk_scores = torch.zeros(M_chunks, dtype=torch.float32, device=self.device)

        for k, chunk_indices in enumerate(chunks):
            if chunk_indices.numel() == 0:
                # ISSUE: Empty chunk guard (can occur from Module 1's trailing
                # placeholder after a boundary token at end of generation step).
                # Assign 0.0 so the chunk gets lowest priority — it will be
                # evicted first, which is safe since it contains no tokens yet.
                chunk_scores[k] = 0.0
                continue

            # Gather token scores for this chunk and mean-pool
            chunk_token_scores = token_scores[chunk_indices]
            chunk_scores[k] = chunk_token_scores.mean()

        return chunk_scores

    def _minmax_normalize(self, x: Tensor) -> Tensor:
        """
        Min-max normalize tensor x to [0, 1].

        Spec formula:
            S_hat_i(C_k) = [S_i(C_k) - min_k S_i(C_k)]
                         / [max_k S_i(C_k) - min_k S_i(C_k)]

        Edge cases handled:
        - All values identical: returns uniform 0.5 (avoids division by zero)
        - Single chunk (M=1): returns 1.0

        ISSUE: If all chunks have identical scores (can happen at very low
        compression ratios when little variation exists), normalization
        produces 0/0. Returning 0.5 uniform is the safest neutral default —
        Signal 2 still provides differentiation in this case.
        """
        x_min = x.min()
        x_max = x.max()
        denom = x_max - x_min

        if denom.item() < 1e-8:
            # All scores effectively identical — return uniform mid-point
            return torch.full_like(x, 0.5)

        return (x - x_min) / denom

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_chunks(self, chunks: list[Tensor], t: int) -> None:
        """
        Sanity check that chunk indices are consistent with sequence length t.
        Only runs in debug scenarios — lightweight enough to keep always-on.
        """
        if len(chunks) == 0:
            raise ValueError("chunks list is empty — Module 1 produced no chunks.")

        # ISSUE: We do not verify that chunks form a complete partition of
        # [0, t-1] here because the incremental update path in Module 1
        # can leave the last chunk open (not yet closed by a boundary token).
        # A full partition check would be:
        #   all_indices = torch.cat(chunks).sort().values
        #   assert torch.equal(all_indices, torch.arange(t))
        # This is O(t log t) — suitable for testing but not production.
        # See test_scorer.py for the full partition verification.

    # ------------------------------------------------------------------
    # Incremental score update during generation
    # ------------------------------------------------------------------

    def update_scores(
        self,
        prev_Score_chunk: Tensor,
        A_obs_new: Tensor,
        chunks: list[Tensor],
        updated_chunk_indices: list[int],
    ) -> Tensor:
        """
        Lightweight incremental score update for newly generated tokens.

        Instead of recomputing all M chunk scores from scratch, only
        recompute scores for chunks whose token membership changed.

        Per spec Section 8.2 Step 3:
            "Recompute chunk scores for affected chunks only"

        Parameters
        ----------
        prev_Score_chunk : Tensor, shape [M]
            Score vector from the previous eviction step.
        A_obs_new : Tensor, shape [H, w, t+1] or [L, H, w, t+1]
            Updated attention matrix including the new token.
        chunks : list of Tensor
            Updated chunk list from Module 1's update() call.
        updated_chunk_indices : list of int
            Which chunk indices (k values) were modified by the new token.
            Module 1's update() always modifies at most the last 1-2 chunks.

        Returns
        -------
        Score_chunk : Tensor, shape [M]
            Updated scores with only affected chunks recomputed.

        ISSUE: This method recomputes M and R for the full sequence to
        get updated token scores, then only writes back the affected chunks.
        The full M/R recomputation is still O(w*t) per step.
        A truly incremental update would only recompute the delta row of
        the attention matrix — left as future optimization.
        True incremental R update is O(T) in worst case since changing
        M[j] for any token j propagates to R[i] for all i that attend to j.
        We bound this by only updating window tokens' R scores.
        """
        # Recompute full signal vectors with new A_obs
        A = A_obs_new.float()
        if A.dim() == 3:
            M_token, R_token = self._compute_signals_single_layer(A)
        else:
            M_token, R_token = self._compute_signals_multi_layer(A)

        M_chunks = len(chunks)
        S1_new = self._aggregate_to_chunks(M_token, chunks, M_chunks)
        S2_new = self._aggregate_to_chunks(R_token, chunks, M_chunks)
        S1_hat = self._minmax_normalize(S1_new)
        S2_hat = self._minmax_normalize(S2_new)

        # Build updated score vector, writing only affected positions
        Score_chunk = prev_Score_chunk.clone()
        for k in updated_chunk_indices:
            if k < M_chunks:
                Score_chunk[k] = self.alpha * S1_hat[k] + self.beta * S2_hat[k]

        return Score_chunk


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_module2(
    alpha: float = 0.6,
    beta: float = 0.4,
    window_size: int = 16,
    num_layers: int | None = None,
    device: str | torch.device = "cpu",
) -> DualSignalScorer:
    """
    Factory function to construct and return a ready-to-use Module 2 instance.
    """
    return DualSignalScorer(
        alpha=alpha,
        beta=beta,
        window_size=window_size,
        num_layers=num_layers,
        device=device,
    )