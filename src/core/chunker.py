"""
TDC-KV Module 1: Sentence-Boundary Chunk Constructor.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

import torch
from torch import Tensor


MIN_CHUNK_TOKENS: int = 5
MAX_CHUNK_TOKENS: int = 64
DEFAULT_BOUNDARY_CHARS: set[str] = {".", ",", "?", "!", ";", ":"}


def build_punctuation_vocab(
    tokenizer: Any,
    boundary_chars: set[str] | None = None,
) -> set[int]:
    """
    Resolve punctuation boundary characters to tokenizer IDs.
    """
    target_chars = boundary_chars or DEFAULT_BOUNDARY_CHARS
    punctuation_token_ids: set[int] = set()

    for token_id in range(tokenizer.vocab_size):
        try:
            surface_form = tokenizer.decode([token_id], skip_special_tokens=True)
            if surface_form.strip() in target_chars:
                punctuation_token_ids.add(token_id)
        except Exception:
            continue

    return punctuation_token_ids


class SentenceBoundaryChunkConstructor:
    """
    Module 1 of TDC-KV.
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        device: str | torch.device = "cpu",
        punct_ids: set[int] | None = None,
    ) -> None:
        """
        Build a chunk constructor from either a tokenizer or explicit punct IDs.
        """
        if punct_ids is None:
            if tokenizer is None:
                raise ValueError("Provide either `tokenizer` or `punct_ids`.")
            punct_ids = build_punctuation_vocab(tokenizer)

        self.min_chunk_tokens = int(min_chunk_tokens)
        self.max_chunk_tokens = int(max_chunk_tokens)
        if self.min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be positive.")
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive.")
        if self.max_chunk_tokens < self.min_chunk_tokens:
            raise ValueError(
                "max_chunk_tokens must be greater than or equal to "
                "min_chunk_tokens."
            )

        self.punct_ids: set[int] = set(punct_ids)
        self.device = torch.device(device)

    def forward(self, X: Tensor) -> tuple[list[Tensor], Tensor]:
        """
        Partition token sequence X into sentence-boundary chunks.
        """
        if X.ndim != 1:
            raise ValueError(f"Expected 1D tensor [t], got shape {tuple(X.shape)}.")

        token_ids = X.to(device=self.device, dtype=torch.long)
        sequence_length = int(token_ids.shape[0])

        if sequence_length == 0:
            empty_map = torch.empty(0, dtype=torch.long, device=self.device)
            return [], empty_map

        boundary_mask = self._compute_boundary_mask(token_ids)
        boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(1)
        raw_chunks = self._build_raw_chunks(boundary_positions, sequence_length)
        merged_chunks = self._merge_small_chunks(raw_chunks)
        chunks = self._split_oversized_chunks(merged_chunks)
        chunk_map = self._build_chunk_map(chunks, sequence_length)
        return chunks, chunk_map

    def _compute_boundary_mask(self, token_ids: Tensor) -> Tensor:
        """
        Returns a boolean tensor of shape [t] where True marks punctuation.
        """
        if len(self.punct_ids) == 0:
            return torch.zeros(token_ids.shape[0], dtype=torch.bool, device=self.device)

        punct_tensor = torch.tensor(
            sorted(self.punct_ids), dtype=torch.long, device=self.device
        )
        return (token_ids.unsqueeze(1) == punct_tensor.unsqueeze(0)).any(dim=1)

    def _build_raw_chunks(
        self, boundary_positions: Tensor, sequence_length: int
    ) -> list[Tensor]:
        """
        Build raw chunks from punctuation boundary positions.
        """
        raw_chunks: list[Tensor] = []
        chunk_start = 0

        for boundary_index in boundary_positions.tolist():
            chunk_end = int(boundary_index)
            chunk_indices = torch.arange(
                chunk_start,
                chunk_end + 1,
                dtype=torch.long,
                device=self.device,
            )
            if chunk_indices.numel() > 0:
                raw_chunks.append(chunk_indices)
            chunk_start = chunk_end + 1

        if chunk_start < sequence_length:
            trailing_indices = torch.arange(
                chunk_start,
                sequence_length,
                dtype=torch.long,
                device=self.device,
            )
            raw_chunks.append(trailing_indices)

        return raw_chunks

    def _merge_small_chunks(self, raw_chunks: list[Tensor]) -> list[Tensor]:
        """
        Merge small chunks into neighboring chunks.
        """
        if self.min_chunk_tokens <= 1 or len(raw_chunks) == 0:
            return raw_chunks

        merged_chunks: list[Tensor] = []
        pending_chunk: Tensor | None = None

        for raw_chunk in raw_chunks:
            current_chunk = raw_chunk
            if pending_chunk is not None:
                current_chunk = torch.cat([pending_chunk, current_chunk])
                pending_chunk = None

            if current_chunk.numel() < self.min_chunk_tokens:
                pending_chunk = current_chunk
            else:
                merged_chunks.append(current_chunk)

        if pending_chunk is not None:
            if len(merged_chunks) > 0:
                merged_chunks[-1] = torch.cat([merged_chunks[-1], pending_chunk])
            else:
                merged_chunks.append(pending_chunk)

        return merged_chunks

    def _split_oversized_chunks(self, chunks: list[Tensor]) -> list[Tensor]:
        """Split long semantic spans while preserving their token order."""
        bounded_chunks: list[Tensor] = []
        for chunk in chunks:
            chunk_length = int(chunk.numel())
            for start in range(0, chunk_length, self.max_chunk_tokens):
                bounded_chunks.append(
                    chunk[start : start + self.max_chunk_tokens]
                )
        return bounded_chunks

    def _build_chunk_map(self, chunks: list[Tensor], sequence_length: int) -> Tensor:
        """
        Build map tensor of shape [t] where map[i] = k means token i in chunk k.
        """
        chunk_map = torch.full(
            (sequence_length,), -1, dtype=torch.long, device=self.device
        )
        for chunk_index, chunk_token_indices in enumerate(chunks):
            chunk_map[chunk_token_indices] = chunk_index

        if torch.any(chunk_map < 0):
            raise RuntimeError(
                "Chunk map construction failed: not all token positions were assigned."
            )
        return chunk_map

    def update(
        self,
        chunks: list[Tensor],
        chunk_map: Tensor,
        new_token_id: int,
        new_position: int,
    ) -> tuple[list[Tensor], Tensor]:
        """
        Incrementally update chunks when a single token is appended.
        """
        if new_position != int(chunk_map.shape[0]):
            raise ValueError(
                "new_position must equal current sequence length (chunk_map.shape[0])."
            )

        normalized_chunks = [
            chunk.to(device=self.device, dtype=torch.long) for chunk in chunks
        ]
        current_map = chunk_map.to(device=self.device, dtype=torch.long)

        new_position_tensor = torch.tensor(
            [new_position], dtype=torch.long, device=self.device
        )
        new_chunk_map = torch.cat(
            [current_map, torch.zeros(1, dtype=torch.long, device=self.device)]
        )

        token_is_boundary = int(new_token_id) in self.punct_ids

        if len(normalized_chunks) == 0:
            normalized_chunks.append(torch.empty(0, dtype=torch.long, device=self.device))
        elif normalized_chunks[-1].numel() >= self.max_chunk_tokens:
            normalized_chunks.append(
                torch.empty(0, dtype=torch.long, device=self.device)
            )

        active_chunk_index = len(normalized_chunks) - 1
        if normalized_chunks[-1].numel() == 0:
            normalized_chunks[-1] = new_position_tensor
        else:
            normalized_chunks[-1] = torch.cat(
                [normalized_chunks[-1], new_position_tensor]
            )
        new_chunk_map[new_position] = active_chunk_index

        chunk_is_full = (
            normalized_chunks[-1].numel() >= self.max_chunk_tokens
        )
        closes_sentence = (
            token_is_boundary
            and normalized_chunks[-1].numel() >= self.min_chunk_tokens
        )
        if chunk_is_full or closes_sentence:
            normalized_chunks.append(
                torch.empty(0, dtype=torch.long, device=self.device)
            )

        return normalized_chunks, new_chunk_map


class FixedSizeChunkConstructor:
    """Deterministic fixed-width constructor for chunking ablations."""

    def __init__(
        self,
        chunk_size: int,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        self.chunk_size = int(chunk_size)
        self.min_chunk_tokens = int(chunk_size)
        self.max_chunk_tokens = int(chunk_size)
        self.device = torch.device(device)

    def forward(self, X: Tensor) -> tuple[list[Tensor], Tensor]:
        if X.ndim != 1:
            raise ValueError(f"Expected 1D tensor [t], got shape {tuple(X.shape)}.")
        sequence_length = int(X.numel())
        chunks = [
            torch.arange(
                start,
                min(start + self.chunk_size, sequence_length),
                dtype=torch.long,
                device=self.device,
            )
            for start in range(0, sequence_length, self.chunk_size)
        ]
        chunk_map = torch.empty(
            sequence_length,
            dtype=torch.long,
            device=self.device,
        )
        for chunk_id, chunk in enumerate(chunks):
            chunk_map[chunk] = chunk_id
        return chunks, chunk_map

    def update(
        self,
        chunks: list[Tensor],
        chunk_map: Tensor,
        new_token_id: int,
        new_position: int,
    ) -> tuple[list[Tensor], Tensor]:
        del new_token_id
        if new_position != int(chunk_map.numel()):
            raise ValueError("new_position must equal the current sequence length.")
        normalized = [chunk.to(self.device, dtype=torch.long) for chunk in chunks]
        if not normalized or int(normalized[-1].numel()) >= self.chunk_size:
            normalized.append(torch.empty(0, dtype=torch.long, device=self.device))
        normalized[-1] = torch.cat(
            [
                normalized[-1],
                torch.tensor([new_position], dtype=torch.long, device=self.device),
            ]
        )
        updated_map = torch.cat(
            [
                chunk_map.to(self.device, dtype=torch.long),
                torch.tensor(
                    [len(normalized) - 1], dtype=torch.long, device=self.device
                ),
            ]
        )
        return normalized, updated_map


def build_module1(
    tokenizer: Any,
    device: str | torch.device = "cpu",
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    max_chunk_tokens: int = MAX_CHUNK_TOKENS,
) -> SentenceBoundaryChunkConstructor:
    """
    Factory function to construct and return a ready-to-use Module 1 instance.
    """
    return SentenceBoundaryChunkConstructor(
        tokenizer=tokenizer,
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        device=device,
    )


def chunk_token_ids(
    token_ids: Iterable[int] | Tensor,
    punct_ids: Iterable[int],
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    max_chunk_tokens: int = MAX_CHUNK_TOKENS,
    device: str | torch.device = "cpu",
) -> tuple[list[list[int]], list[int]]:
    """
    Notebook/script-friendly helper that returns Python lists.
    """
    if isinstance(token_ids, Tensor):
        token_tensor = token_ids.to(device=device, dtype=torch.long)
    else:
        token_tensor = torch.tensor(list(token_ids), dtype=torch.long, device=device)

    constructor = SentenceBoundaryChunkConstructor(
        tokenizer=None,
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        device=device,
        punct_ids=set(punct_ids),
    )
    chunk_tensors, chunk_map_tensor = constructor.forward(token_tensor)
    chunk_lists = [chunk.detach().cpu().tolist() for chunk in chunk_tensors]
    chunk_map = chunk_map_tensor.detach().cpu().tolist()
    return chunk_lists, chunk_map


def _parse_csv_ints(raw: str) -> list[int]:
    text = raw.strip()
    if text == "":
        return []
    return [int(piece.strip()) for piece in text.split(",") if piece.strip() != ""]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Module 1 chunking on a comma-separated token ID list."
    )
    parser.add_argument(
        "--token-ids",
        required=True,
        help='Comma-separated token IDs, for example: "11,45,78,29913,2".',
    )
    punct_source = parser.add_mutually_exclusive_group(required=True)
    punct_source.add_argument(
        "--punct-ids",
        help='Comma-separated punctuation token IDs, for example: "13,30,2".',
    )
    punct_source.add_argument(
        "--tokenizer-name",
        help="HuggingFace tokenizer name/path used to derive punctuation token IDs.",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=MIN_CHUNK_TOKENS,
        help="Minimum chunk size before merge behavior is applied.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=MAX_CHUNK_TOKENS,
        help="Maximum chunk size after sentence-boundary construction.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device string, for example: "cpu" or "cuda".',
    )
    args = parser.parse_args(argv)

    token_ids = _parse_csv_ints(args.token_ids)
    if args.punct_ids is not None:
        punct_ids = set(_parse_csv_ints(args.punct_ids))
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        punct_ids = build_punctuation_vocab(tokenizer)

    chunks, chunk_map = chunk_token_ids(
        token_ids=token_ids,
        punct_ids=punct_ids,
        min_chunk_tokens=args.min_chunk_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        device=args.device,
    )
    payload = {
        "token_ids": token_ids,
        "punct_ids": sorted(punct_ids),
        "min_chunk_tokens": args.min_chunk_tokens,
        "max_chunk_tokens": args.max_chunk_tokens,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "map": chunk_map,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
