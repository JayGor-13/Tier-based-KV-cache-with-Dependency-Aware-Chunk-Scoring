import json
import subprocess
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if torch is not None:
    from src.core.chunker import (
        SentenceBoundaryChunkConstructor,
        build_punctuation_vocab,
        chunk_token_ids,
    )
else:  # pragma: no cover
    SentenceBoundaryChunkConstructor = None  # type: ignore[assignment]
    build_punctuation_vocab = None  # type: ignore[assignment]
    chunk_token_ids = None  # type: ignore[assignment]


class DummyTokenizer:
    def __init__(self, id_to_surface):
        self.id_to_surface = dict(id_to_surface)
        self.vocab_size = (max(id_to_surface.keys()) + 1) if id_to_surface else 0

    def decode(self, token_ids, skip_special_tokens=True):
        token_id = token_ids[0]
        return self.id_to_surface.get(token_id, "x")


def _to_lists(chunks):
    return [chunk.tolist() for chunk in chunks]


class TestChunkerModule1(unittest.TestCase):
    def setUp(self):
        if torch is None:  # pragma: no cover
            self.skipTest("torch is required to run chunker tests")

    def test_build_punctuation_vocab_includes_spec_chars_and_strip_behavior(self):
        tokenizer = DummyTokenizer(
            {
                0: "alpha",
                1: ".",
                2: " ,",
                3: "?",
                4: " !",
                5: ";",
                6: ":",
                7: "..",
            }
        )

        punct_ids = build_punctuation_vocab(tokenizer)
        self.assertEqual(punct_ids, {1, 2, 3, 4, 5, 6})

    def test_forward_partitions_indices_without_gaps_or_overlap(self):
        constructor = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids={2, 3},
            min_chunk_tokens=1,
        )
        token_ids = torch.tensor([10, 11, 2, 12, 3, 13], dtype=torch.long)

        chunks, chunk_map = constructor.forward(token_ids)

        self.assertEqual(_to_lists(chunks), [[0, 1, 2], [3, 4], [5]])
        self.assertEqual(chunk_map.tolist(), [0, 0, 0, 1, 1, 2])

    def test_forward_without_boundaries_returns_single_chunk(self):
        constructor = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids={999},
            min_chunk_tokens=1,
        )
        token_ids = torch.tensor([10, 11, 12], dtype=torch.long)

        chunks, chunk_map = constructor.forward(token_ids)

        self.assertEqual(_to_lists(chunks), [[0, 1, 2]])
        self.assertEqual(chunk_map.tolist(), [0, 0, 0])

    def test_forward_with_empty_input_returns_empty_outputs(self):
        constructor = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids={1},
            min_chunk_tokens=1,
        )
        token_ids = torch.tensor([], dtype=torch.long)

        chunks, chunk_map = constructor.forward(token_ids)

        self.assertEqual(chunks, [])
        self.assertEqual(chunk_map.tolist(), [])

    def test_merge_small_chunks_respects_min_chunk_tokens(self):
        constructor = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids={2, 3},
            min_chunk_tokens=3,
        )
        token_ids = torch.tensor([10, 2, 11, 3, 12, 13], dtype=torch.long)

        chunks, chunk_map = constructor.forward(token_ids)

        # The tail chunk [4, 5] is below threshold and is merged backward.
        self.assertEqual(_to_lists(chunks), [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(chunk_map.tolist(), [0, 0, 0, 0, 0, 0])

    def test_update_non_boundary_appends_to_active_chunk(self):
        constructor = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids={9},
            min_chunk_tokens=1,
        )
        chunks = [torch.tensor([0, 1], dtype=torch.long)]
        chunk_map = torch.tensor([0, 0], dtype=torch.long)

        updated_chunks, updated_map = constructor.update(
            chunks=chunks,
            chunk_map=chunk_map,
            new_token_id=7,
            new_position=2,
        )

        self.assertEqual(_to_lists(updated_chunks), [[0, 1, 2]])
        self.assertEqual(updated_map.tolist(), [0, 0, 0])

    def test_update_boundary_then_non_boundary_flow(self):
        constructor = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids={9},
            min_chunk_tokens=1,
        )
        chunks = [torch.tensor([0, 1], dtype=torch.long)]
        chunk_map = torch.tensor([0, 0], dtype=torch.long)

        chunks, chunk_map = constructor.update(
            chunks=chunks,
            chunk_map=chunk_map,
            new_token_id=9,
            new_position=2,
        )
        self.assertEqual(_to_lists(chunks), [[0, 1, 2], []])
        self.assertEqual(chunk_map.tolist(), [0, 0, 0])

        chunks, chunk_map = constructor.update(
            chunks=chunks,
            chunk_map=chunk_map,
            new_token_id=7,
            new_position=3,
        )
        self.assertEqual(_to_lists(chunks), [[0, 1, 2], [3]])
        self.assertEqual(chunk_map.tolist(), [0, 0, 0, 1])

    def test_update_validates_new_position_matches_sequence_length(self):
        constructor = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids={9},
            min_chunk_tokens=1,
        )

        with self.assertRaises(ValueError):
            constructor.update(
                chunks=[torch.tensor([0], dtype=torch.long)],
                chunk_map=torch.tensor([0], dtype=torch.long),
                new_token_id=7,
                new_position=3,
            )

    def test_chunk_token_ids_helper_returns_python_lists(self):
        chunks, chunk_map = chunk_token_ids(
            token_ids=[10, 11, 2, 12],
            punct_ids={2},
            min_chunk_tokens=1,
        )

        self.assertEqual(chunks, [[0, 1, 2], [3]])
        self.assertEqual(chunk_map, [0, 0, 0, 1])

    def test_chunker_cli_smoke(self):
        command = [
            sys.executable,
            "-m",
            "src.core.chunker",
            "--token-ids",
            "10,11,2,12",
            "--punct-ids",
            "2",
            "--min-chunk-tokens",
            "1",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["chunks"], [[0, 1, 2], [3]])
        self.assertEqual(payload["map"], [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
