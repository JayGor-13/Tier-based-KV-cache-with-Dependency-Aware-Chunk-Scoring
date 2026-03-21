"""
Unit tests for src/core/scorer.py (Module 2: DualSignalScorer).

Run:
    python -m unittest tests.test_scorer -v
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

DualSignalScorer = None
build_module2 = None
SCORER_IMPORT_ERROR: str | None = None

if torch is not None:
    try:
        from src.core.scorer import DualSignalScorer, build_module2
    except Exception as primary_exc:  # pragma: no cover
        # Fallback: import scorer.py directly, avoiding package-level side effects.
        try:
            scorer_path = PROJECT_ROOT / "src" / "core" / "scorer.py"
            spec = importlib.util.spec_from_file_location("tdckv_scorer", scorer_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to load scorer module spec")
            scorer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scorer_module)
            DualSignalScorer = scorer_module.DualSignalScorer
            build_module2 = scorer_module.build_module2
        except Exception as fallback_exc:
            SCORER_IMPORT_ERROR = (
                f"primary import failed: {primary_exc}; "
                f"fallback import failed: {fallback_exc}"
            )


def _make_attention(h: int, w: int, t: int, seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn(h, w, t, dtype=torch.float32)
    return torch.softmax(logits, dim=-1)


def _make_attention_multilayer(
    layers: int, h: int, w: int, t: int, seed: int = 0
):
    torch.manual_seed(seed)
    logits = torch.randn(layers, h, w, t, dtype=torch.float32)
    return torch.softmax(logits, dim=-1)


def _uniform_chunks(t: int, chunk_size: int):
    chunks = []
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        chunks.append(torch.arange(start, end, dtype=torch.long))
    return chunks


class TestDualSignalScorer(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:  # pragma: no cover
            self.skipTest("torch is required to run scorer tests")
        if DualSignalScorer is None or build_module2 is None:  # pragma: no cover
            self.skipTest(f"Unable to import scorer module: {SCORER_IMPORT_ERROR}")

    def test_alpha_beta_constraint(self) -> None:
        with self.assertRaises(ValueError):
            DualSignalScorer(alpha=0.9, beta=0.2)

    def test_forward_single_layer_shape_range_and_dtype(self) -> None:
        h, w, t = 4, 8, 32
        a_obs = _make_attention(h, w, t, seed=10)
        chunks = _uniform_chunks(t, chunk_size=4)

        scorer = build_module2(alpha=0.6, beta=0.4, window_size=w, device="cpu")
        scores = scorer.forward(a_obs, chunks)

        self.assertEqual(tuple(scores.shape), (len(chunks),))
        self.assertEqual(scores.dtype, torch.float32)
        self.assertGreaterEqual(scores.min().item(), -1e-6)
        self.assertLessEqual(scores.max().item(), 1.0 + 1e-6)
        self.assertFalse(torch.isnan(scores).any().item())
        self.assertFalse(torch.isinf(scores).any().item())

    def test_forward_multi_layer_shape_and_range(self) -> None:
        layers, h, w, t = 3, 2, 6, 24
        a_obs = _make_attention_multilayer(layers, h, w, t, seed=20)
        chunks = _uniform_chunks(t, chunk_size=6)

        scorer = build_module2(window_size=w, num_layers=layers, device="cpu")
        scores = scorer.forward(a_obs, chunks)

        self.assertEqual(tuple(scores.shape), (len(chunks),))
        self.assertGreaterEqual(scores.min().item(), -1e-6)
        self.assertLessEqual(scores.max().item(), 1.0 + 1e-6)

    def test_forward_rejects_invalid_attention_rank(self) -> None:
        scorer = build_module2(window_size=4, device="cpu")
        bad_a_obs = torch.randn(4, 10, dtype=torch.float32)  # rank 2, invalid
        chunks = _uniform_chunks(t=10, chunk_size=5)

        with self.assertRaises(ValueError):
            _ = scorer.forward(bad_a_obs, chunks)

    def test_forward_rejects_empty_chunk_list(self) -> None:
        scorer = build_module2(window_size=4, device="cpu")
        a_obs = _make_attention(h=2, w=4, t=12, seed=1)

        with self.assertRaises(ValueError):
            _ = scorer.forward(a_obs, [])

    def test_uniform_attention_produces_uniform_scores(self) -> None:
        h, w, t = 2, 4, 20
        a_obs = torch.full((h, w, t), 1.0 / t, dtype=torch.float32)
        chunks = _uniform_chunks(t, chunk_size=5)

        scorer = build_module2(window_size=w, device="cpu")
        scores = scorer.forward(a_obs, chunks)

        self.assertTrue(torch.allclose(scores, scores[0].expand_as(scores), atol=1e-6))

    def test_single_chunk_returns_half_due_to_constant_normalization(self) -> None:
        h, w, t = 2, 4, 10
        a_obs = _make_attention(h, w, t, seed=3)
        chunks = [torch.arange(t, dtype=torch.long)]

        scorer = build_module2(window_size=w, device="cpu")
        scores = scorer.forward(a_obs, chunks)

        self.assertEqual(tuple(scores.shape), (1,))
        self.assertTrue(torch.allclose(scores, torch.tensor([0.5]), atol=1e-6))

    def test_empty_chunk_is_handled_without_nan(self) -> None:
        h, w, t = 2, 4, 16
        a_obs = _make_attention(h, w, t, seed=5)
        chunks = _uniform_chunks(t, chunk_size=4)
        chunks.append(torch.empty(0, dtype=torch.long))

        scorer = build_module2(window_size=w, device="cpu")
        scores = scorer.forward(a_obs, chunks)

        self.assertEqual(tuple(scores.shape), (len(chunks),))
        self.assertFalse(torch.isnan(scores).any().item())
        self.assertAlmostEqual(scores[-1].item(), scores.min().item(), places=6)

    def test_layer_weights_sum_to_one(self) -> None:
        scorer = DualSignalScorer(num_layers=8, device="cpu")
        self.assertIsNotNone(scorer.layer_weights)
        self.assertAlmostEqual(float(scorer.layer_weights.sum().item()), 1.0, places=6)

    def test_signal2_defaults_to_m_outside_window(self) -> None:
        scorer = build_module2(window_size=2, device="cpu")

        # Shape [H=1, w=2, t=5], so out-of-window positions are [0,1,2].
        a_obs = torch.zeros(1, 2, 5, dtype=torch.float32)
        a_obs[0, 0, 0] = 1.0
        a_obs[0, 1, 1] = 1.0

        m_token, r_token = scorer._compute_signals_single_layer(a_obs)
        self.assertTrue(torch.allclose(r_token[:3], m_token[:3], atol=1e-6))

    def test_update_scores_only_changes_requested_chunk_indices(self) -> None:
        h, w, t = 2, 4, 20
        chunks = _uniform_chunks(t, chunk_size=5)
        scorer = build_module2(window_size=w, device="cpu")

        a_obs_old = _make_attention(h, w, t, seed=11)
        prev_scores = scorer.forward(a_obs_old, chunks)

        a_obs_new = _make_attention(h, w, t, seed=12)
        updated_scores = scorer.update_scores(
            prev_Score_chunk=prev_scores,
            A_obs_new=a_obs_new,
            chunks=chunks,
            updated_chunk_indices=[1, 3],
        )

        self.assertTrue(torch.equal(updated_scores[[0, 2]], prev_scores[[0, 2]]))
        delta = torch.abs(updated_scores[[1, 3]] - prev_scores[[1, 3]])
        self.assertTrue(torch.any(delta > 1e-8).item())

    def test_update_scores_ignores_out_of_range_chunk_indices(self) -> None:
        h, w, t = 2, 4, 20
        chunks = _uniform_chunks(t, chunk_size=5)
        scorer = build_module2(window_size=w, device="cpu")

        a_obs_old = _make_attention(h, w, t, seed=21)
        prev_scores = scorer.forward(a_obs_old, chunks)

        a_obs_new = _make_attention(h, w, t, seed=22)
        updated_scores = scorer.update_scores(
            prev_Score_chunk=prev_scores,
            A_obs_new=a_obs_new,
            chunks=chunks,
            updated_chunk_indices=[999],
        )

        self.assertTrue(torch.equal(updated_scores, prev_scores))


if __name__ == "__main__":
    unittest.main(verbosity=2)
