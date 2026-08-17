import torch

from src.core.chunker import FixedSizeChunkConstructor


def test_fixed_size_constructor_supports_chunk_ablation():
    constructor = FixedSizeChunkConstructor(3)
    chunks, chunk_map = constructor.forward(torch.arange(8))
    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert chunk_map.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
