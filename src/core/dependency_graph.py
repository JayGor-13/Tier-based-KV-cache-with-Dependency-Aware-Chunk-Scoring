"""Sparse chunk-dependency graph construction and relevance routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SparseChunkDependencyGraph:
    """Top-k outgoing dependency edges for each query chunk.

    ``neighbor_indices[q, e]`` identifies a key chunk that query chunk ``q``
    attended to during prefill. ``edge_weights`` stores the corresponding
    normalized attention mass. Missing edges use index ``-1`` and weight zero.
    """

    neighbor_indices: torch.Tensor
    edge_weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.neighbor_indices.ndim != 2 or self.edge_weights.ndim != 2:
            raise ValueError("Dependency graph tensors must have shape [chunks, top_k].")
        if self.neighbor_indices.shape != self.edge_weights.shape:
            raise ValueError("Dependency graph indices and weights must have matching shapes.")
        if self.neighbor_indices.dtype != torch.long:
            raise ValueError("Dependency graph indices must use torch.long dtype.")
        if not torch.is_floating_point(self.edge_weights):
            raise ValueError("Dependency graph weights must be floating point.")
        if bool((self.edge_weights < 0).any().item()):
            raise ValueError("Dependency graph weights must be non-negative.")

    @property
    def num_chunks(self) -> int:
        return int(self.neighbor_indices.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.neighbor_indices.shape[1])

    def to(self, device: str | torch.device) -> "SparseChunkDependencyGraph":
        target = torch.device(device)
        return SparseChunkDependencyGraph(
            neighbor_indices=self.neighbor_indices.to(target),
            edge_weights=self.edge_weights.to(target),
        )

    def route(self, chunk_relevance: torch.Tensor) -> torch.Tensor:
        """Propagate current relevance backward to historical dependencies.

        Graph rows are causal query-to-key edges: ``q -> r`` means chunk ``q``
        attended to the older chunk ``r`` during prefill.  A currently relevant
        chunk therefore lends relevance to the historical chunks on which it
        depended.  The direct relevance is retained as a residual so a chunk
        never loses its own evidence merely because it also has incoming edges.
        """
        if chunk_relevance.ndim != 1:
            raise ValueError("chunk_relevance must be a 1D tensor.")
        if chunk_relevance.numel() != self.num_chunks:
            raise ValueError(
                "chunk_relevance length must match dependency graph chunk count."
            )

        relevance = chunk_relevance.to(dtype=torch.float32)
        indices = self.neighbor_indices.to(device=relevance.device)
        weights = self.edge_weights.to(device=relevance.device, dtype=torch.float32)
        valid = indices >= 0

        if self.top_k == 0:
            return relevance.clone()

        valid_weights = torch.where(valid, weights, torch.zeros_like(weights))
        query_ids = torch.arange(
            self.num_chunks,
            dtype=torch.long,
            device=relevance.device,
        ).unsqueeze(1).expand_as(indices)
        contributions = relevance[query_ids] * valid_weights

        routed = relevance.clone()
        routed.scatter_add_(
            0,
            indices[valid],
            contributions[valid],
        )
        return routed


class SparseChunkDependencyGraphBuilder:
    """Incrementally aggregate attention rows into a sparse chunk graph."""

    def __init__(
        self,
        *,
        num_chunks: int,
        top_k: int = 8,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_chunks < 0:
            raise ValueError("num_chunks must be non-negative.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")

        self.num_chunks = int(num_chunks)
        self.top_k = min(int(top_k), max(self.num_chunks - 1, 0))
        self.device = torch.device(device)
        self._neighbor_indices = torch.full(
            (self.num_chunks, self.top_k),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._edge_weights = torch.zeros(
            (self.num_chunks, self.top_k),
            dtype=torch.float32,
            device=self.device,
        )
        self._active_chunk: int | None = None
        self._active_mass = torch.zeros(
            self.num_chunks, dtype=torch.float32, device=self.device
        )
        self._active_query_count = 0
        self._last_finalized_chunk = -1

    def update(
        self,
        attention_rows: torch.Tensor,
        *,
        query_chunk_ids: torch.Tensor,
        key_chunk_ids: torch.Tensor,
    ) -> None:
        """Consume attention rows in non-decreasing query-chunk order.

        ``attention_rows`` may be ``[queries, keys]`` after head/layer
        aggregation or ``[heads, queries, keys]``. Head rows are averaged.
        """
        rows = attention_rows.to(device=self.device, dtype=torch.float32)
        if rows.ndim == 3:
            rows = rows.mean(dim=0)
        if rows.ndim != 2:
            raise ValueError("attention_rows must have shape [Q,K] or [H,Q,K].")

        query_ids = query_chunk_ids.to(device=self.device, dtype=torch.long)
        key_ids = key_chunk_ids.to(device=self.device, dtype=torch.long)
        if query_ids.ndim != 1 or query_ids.numel() != rows.shape[0]:
            raise ValueError("query_chunk_ids must have shape [Q].")
        if key_ids.ndim != 1 or key_ids.numel() != rows.shape[1]:
            raise ValueError("key_chunk_ids must have shape [K].")
        self._validate_chunk_ids(query_ids, "query_chunk_ids")
        self._validate_chunk_ids(key_ids, "key_chunk_ids")
        if query_ids.numel() > 1 and bool((query_ids[1:] < query_ids[:-1]).any().item()):
            raise ValueError("query_chunk_ids must be non-decreasing.")

        for chunk_id_tensor in torch.unique_consecutive(query_ids):
            chunk_id = int(chunk_id_tensor.item())
            if chunk_id <= self._last_finalized_chunk:
                raise ValueError("A finalized query chunk cannot be updated again.")
            if self._active_chunk != chunk_id:
                self._finalize_active_chunk()
                self._active_chunk = chunk_id

            local_mask = query_ids == chunk_id
            local_rows = rows[local_mask]
            token_mass = local_rows.sum(dim=0)
            chunk_mass = torch.zeros(
                self.num_chunks, dtype=torch.float32, device=self.device
            )
            chunk_mass.scatter_add_(0, key_ids, token_mass)
            self._active_mass += chunk_mass
            self._active_query_count += int(local_rows.shape[0])

    def finalize(self) -> SparseChunkDependencyGraph:
        self._finalize_active_chunk()
        return SparseChunkDependencyGraph(
            neighbor_indices=self._neighbor_indices,
            edge_weights=self._edge_weights,
        )

    def _validate_chunk_ids(self, chunk_ids: torch.Tensor, name: str) -> None:
        if chunk_ids.numel() == 0:
            return
        low = int(chunk_ids.min().item())
        high = int(chunk_ids.max().item())
        if low < 0 or high >= self.num_chunks:
            raise IndexError(
                f"{name} contains chunk ids [{low}, {high}] for "
                f"{self.num_chunks} chunks."
            )

    def _finalize_active_chunk(self) -> None:
        if self._active_chunk is None:
            return

        chunk_id = self._active_chunk
        if self._active_query_count > 0 and self.top_k > 0:
            mass = self._active_mass / float(self._active_query_count)
            mass[chunk_id] = 0.0
            positive_count = int((mass > 0).sum().item())
            edge_count = min(self.top_k, positive_count)
            if edge_count > 0:
                values, indices = torch.topk(mass, k=edge_count, largest=True)
                values = values / values.sum().clamp_min(torch.finfo(torch.float32).eps)
                self._neighbor_indices[chunk_id, :edge_count] = indices
                self._edge_weights[chunk_id, :edge_count] = values

        self._last_finalized_chunk = chunk_id
        self._active_chunk = None
        self._active_mass.zero_()
        self._active_query_count = 0


def aggregate_attention_rows(
    attentions: Sequence[torch.Tensor],
    *,
    mode: str = "last",
    layer_index: int = -1,
    layer_weighting: str = "linear",
) -> torch.Tensor:
    """Aggregate HF attention layers and heads into ``[queries, keys]`` rows."""
    if not attentions:
        raise ValueError("Model output did not include attentions.")

    mode = mode.lower()
    layer_weighting = str(layer_weighting).strip().lower()
    if layer_weighting not in {"linear", "uniform"}:
        raise ValueError("layer_weighting must be `linear` or `uniform`.")
    layer_count = len(attentions)
    if mode == "last":
        idx = int(layer_index)
        if idx < 0:
            idx += layer_count
        if idx < 0 or idx >= layer_count:
            raise IndexError(f"layer_index {layer_index} is out of range.")
        selected = [(attentions[idx], 1.0)]
    elif mode == "all":
        raw_weights = (
            torch.arange(1, layer_count + 1, dtype=torch.float32)
            if layer_weighting == "linear"
            else torch.ones(layer_count, dtype=torch.float32)
        )
        raw_weights /= raw_weights.sum()
        selected = list(zip(attentions, raw_weights.tolist()))
    else:
        raise ValueError("mode must be `last` or `all`.")

    combined_rows: torch.Tensor | None = None
    for attention, weight in selected:
        if attention is None:
            raise ValueError(
                "A selected attention layer is missing. Use an attention "
                "implementation that supports output_attentions=True."
            )
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(
                "Expected attention shape [1,heads,queries,keys] for graph construction."
            )
        layer_rows = attention[0].detach().to(torch.float32).mean(dim=0)
        if combined_rows is None:
            combined_rows = float(weight) * layer_rows
        else:
            if layer_rows.shape != combined_rows.shape:
                raise ValueError("All selected attention layers must have matching shapes.")
            combined_rows += float(weight) * layer_rows

    assert combined_rows is not None
    return combined_rows


def build_sparse_chunk_dependency_graph(
    attentions: Sequence[torch.Tensor],
    *,
    chunk_map: torch.Tensor,
    top_k: int = 8,
    mode: str = "last",
    layer_index: int = -1,
    layer_weighting: str = "linear",
    offload_to_cpu: bool = True,
) -> SparseChunkDependencyGraph:
    """Build a sparse graph from model attention outputs.

    This adapter consumes the full-prefill attention format used by Hugging Face.
    The builder itself is incremental and can also consume blockwise attention
    rows when the prefill path is converted to bounded-memory collection.
    """
    if chunk_map.ndim != 1 or chunk_map.numel() == 0:
        raise ValueError("chunk_map must be a non-empty 1D tensor.")

    combined_rows = aggregate_attention_rows(
        attentions,
        mode=mode,
        layer_index=layer_index,
        layer_weighting=layer_weighting,
    )
    query_len, key_len = combined_rows.shape
    if query_len > chunk_map.numel() or key_len > chunk_map.numel():
        raise ValueError("Attention axes exceed the supplied chunk map length.")

    graph_device = combined_rows.device
    map_on_device = chunk_map.to(device=graph_device, dtype=torch.long)
    num_chunks = int(map_on_device.max().item()) + 1
    builder = SparseChunkDependencyGraphBuilder(
        num_chunks=num_chunks, top_k=top_k, device=graph_device
    )
    builder.update(
        combined_rows,
        query_chunk_ids=map_on_device[:query_len],
        key_chunk_ids=map_on_device[:key_len],
    )
    graph = builder.finalize()
    return graph.to("cpu") if offload_to_cpu else graph


__all__ = [
    "SparseChunkDependencyGraph",
    "SparseChunkDependencyGraphBuilder",
    "aggregate_attention_rows",
    "build_sparse_chunk_dependency_graph",
]
