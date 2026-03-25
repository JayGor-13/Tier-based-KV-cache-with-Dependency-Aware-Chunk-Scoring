"""TDC-KV Module 2: dual-signal chunk scorer."""

from __future__ import annotations

import torch
from torch import Tensor


class DualSignalScorer:
    """Compute chunk scores from observed attention via two token-level signals.

    Signal 1 (M): attention mass from observation-window queries.
    Signal 2 (R): forward routing computed only for observation-window tokens.

    For non-window tokens, `R` fallback can be:
    - `"mass"`: use current-step `M` directly.
    - `"ema"`: use running EMA state (updated each call) for cheap temporal smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        window_size: int = 16,
        num_layers: int | None = None,
        device: str | torch.device = "cpu",
        r_fallback_mode: str = "ema",
        ema_decay: float = 0.9,
    ) -> None:
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(
                f"alpha + beta must equal 1.0, got {alpha} + {beta} = {alpha + beta}"
            )
        if r_fallback_mode not in {"ema", "mass"}:
            raise ValueError("r_fallback_mode must be one of {'ema', 'mass'}.")
        if not (0.0 <= float(ema_decay) < 1.0):
            raise ValueError("ema_decay must satisfy 0.0 <= ema_decay < 1.0.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.window_size = int(window_size)
        self.num_layers = num_layers
        self.device = torch.device(device)
        self.r_fallback_mode = r_fallback_mode
        self.ema_decay = float(ema_decay)
        self._r_ema_state: Tensor | None = None

        if num_layers is not None and num_layers > 0:
            layer_indices = torch.arange(1, num_layers + 1, dtype=torch.float32)
            self.layer_weights: Tensor | None = (layer_indices / layer_indices.sum()).to(
                self.device
            )
        else:
            self.layer_weights = None

    def reset_state(self) -> None:
        """Reset internal EMA fallback state."""
        self._r_ema_state = None

    def forward(self, A_obs: Tensor, chunks: list[Tensor]) -> Tensor:
        """Return fused chunk importance scores with shape `[num_chunks]`."""
        A = A_obs.float()
        if A.dim() == 3:
            M_token, R_token = self._compute_signals_single_layer(A)
        elif A.dim() == 4:
            M_token, R_token = self._compute_signals_multi_layer(A)
        else:
            raise ValueError(
                f"A_obs must be 3D [H,w,t] or 4D [L,H,w,t], got shape {tuple(A_obs.shape)}"
            )

        t = int(M_token.shape[0])
        M_chunks = len(chunks)
        self._validate_chunks(chunks, t)

        S1 = self._aggregate_to_chunks(M_token, chunks, M_chunks)
        S2 = self._aggregate_to_chunks(R_token, chunks, M_chunks)

        S1_hat = self._minmax_normalize(S1)
        S2_hat = self._minmax_normalize(S2)
        return self.alpha * S1_hat + self.beta * S2_hat

    def _fallback_r_vector(self, M_token: Tensor) -> Tensor:
        """Build non-window fallback vector for R with optional EMA state."""
        if self.r_fallback_mode == "mass":
            return M_token.clone()

        if self._r_ema_state is None or self._r_ema_state.shape != M_token.shape:
            fallback = M_token.clone()
        else:
            fallback = self.ema_decay * self._r_ema_state + (1.0 - self.ema_decay) * M_token
        return fallback

    def _commit_r_state(self, R_token: Tensor) -> None:
        if self.r_fallback_mode == "ema":
            self._r_ema_state = R_token.detach().clone()

    def _compute_signals_single_layer(self, A: Tensor) -> tuple[Tensor, Tensor]:
        """Compute `(M, R)` from single-layer observed attention `[H,w,t]`."""
        _, w, t = A.shape

        # Signal 1: attention mass over heads + observation-window queries.
        M_token = A.sum(dim=(0, 1))  # [t]

        # Signal 2: routing only for window tokens; fallback for others.
        R_token = self._fallback_r_vector(M_token)
        window_start = max(t - w, 0)

        # R_window[q] = mean_h sum_i A[h,q,i] * M[i], for q in observation window.
        # Using head-mean avoids artificial head-count amplification.
        R_window = (A * M_token[None, None, :]).sum(dim=2).mean(dim=0)  # [w]
        actual_window_len = t - window_start
        R_token[window_start:t] = R_window[:actual_window_len]
        self._commit_r_state(R_token)
        return M_token, R_token

    def _compute_signals_multi_layer(self, A: Tensor) -> tuple[Tensor, Tensor]:
        """Compute layer-weighted `(M, R)` from `[L,H,w,t]` without Python loops."""
        L, _, w, t = A.shape
        if self.layer_weights is not None and self.layer_weights.shape[0] == L:
            layer_w = self.layer_weights
        else:
            layer_w = torch.full((L,), 1.0 / float(L), dtype=torch.float32, device=self.device)

        # Per-layer mass: [L,t]
        M_per_layer = A.sum(dim=(1, 2))
        M_token = (layer_w[:, None] * M_per_layer).sum(dim=0)  # [t]

        R_token = self._fallback_r_vector(M_token)
        window_start = max(t - w, 0)

        # Per-layer routing window: [L,w]
        Rw_per_layer = (A * M_per_layer[:, None, None, :]).sum(dim=3).mean(dim=1)
        R_window = (layer_w[:, None] * Rw_per_layer).sum(dim=0)  # [w]
        actual_window_len = t - window_start
        R_token[window_start:t] = R_window[:actual_window_len]
        self._commit_r_state(R_token)
        return M_token, R_token

    def _aggregate_to_chunks(
        self,
        token_scores: Tensor,
        chunks: list[Tensor],
        M_chunks: int,
    ) -> Tensor:
        chunk_scores = torch.zeros(M_chunks, dtype=torch.float32, device=self.device)
        for k, chunk_indices in enumerate(chunks):
            if chunk_indices.numel() == 0:
                chunk_scores[k] = 0.0
                continue
            chunk_scores[k] = token_scores[chunk_indices].mean()
        return chunk_scores

    def _minmax_normalize(self, x: Tensor) -> Tensor:
        x_min = x.min()
        x_max = x.max()
        denom = x_max - x_min
        if denom.item() < 1e-8:
            return torch.full_like(x, 0.5)
        return (x - x_min) / denom

    def _validate_chunks(self, chunks: list[Tensor], t: int) -> None:
        if len(chunks) == 0:
            raise ValueError("chunks list is empty - Module 1 produced no chunks.")
        del t

    def update_scores(
        self,
        prev_Score_chunk: Tensor,
        A_obs_new: Tensor,
        chunks: list[Tensor],
        updated_chunk_indices: list[int],
    ) -> Tensor:
        """Recompute token signals, then update only affected chunk indices."""
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

        score_chunk = prev_Score_chunk.clone()
        for k in updated_chunk_indices:
            if 0 <= int(k) < M_chunks:
                score_chunk[k] = self.alpha * S1_hat[k] + self.beta * S2_hat[k]
        return score_chunk


def build_module2(
    alpha: float = 0.6,
    beta: float = 0.4,
    window_size: int = 16,
    num_layers: int | None = None,
    device: str | torch.device = "cpu",
    r_fallback_mode: str = "ema",
    ema_decay: float = 0.9,
) -> DualSignalScorer:
    """Factory helper for Module 2 scorer."""
    return DualSignalScorer(
        alpha=alpha,
        beta=beta,
        window_size=window_size,
        num_layers=num_layers,
        device=device,
        r_fallback_mode=r_fallback_mode,
        ema_decay=ema_decay,
    )
