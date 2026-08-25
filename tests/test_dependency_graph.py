import torch

from src.core.dependency_graph import (
    SparseChunkDependencyGraph,
    SparseChunkDependencyGraphBuilder,
    aggregate_attention_rows,
    build_sparse_chunk_dependency_graph,
)
from src.core.scorer import DualSignalScorer


def test_incremental_builder_keeps_top_inter_chunk_edges():
    rows = torch.zeros(6, 6, dtype=torch.float32)
    rows[0, 0] = 1.0
    rows[1, 1] = 1.0
    rows[2:4, 0:2] = 0.4
    rows[2:4, 2:4] = 0.1
    rows[4:6, 0:2] = 0.05
    rows[4:6, 2:4] = 0.45
    chunk_map = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    builder = SparseChunkDependencyGraphBuilder(num_chunks=3, top_k=1)
    builder.update(
        rows[:3],
        query_chunk_ids=chunk_map[:3],
        key_chunk_ids=chunk_map,
    )
    builder.update(
        rows[3:],
        query_chunk_ids=chunk_map[3:],
        key_chunk_ids=chunk_map,
    )
    graph = builder.finalize()

    assert graph.neighbor_indices.tolist() == [[-1], [0], [1]]
    assert torch.allclose(graph.edge_weights[1:], torch.ones(2, 1))


def test_graph_routes_relevance_to_historical_dependencies():
    graph = SparseChunkDependencyGraph(
        neighbor_indices=torch.tensor([[-1], [0], [1]], dtype=torch.long),
        edge_weights=torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32),
    )

    routed = graph.route(torch.tensor([1.0, 0.2, 0.1]))

    assert torch.allclose(routed, torch.tensor([1.2, 0.3, 0.1]))


def test_routed_signal_changes_old_chunk_ranking():
    attention_obs = torch.tensor(
        [
            [
                [0.6, 0.2, 0.1, 0.1, 0.0],
                [0.6, 0.2, 0.1, 0.0, 0.1],
            ]
        ],
        dtype=torch.float32,
    )
    chunks = [torch.tensor([i], dtype=torch.long) for i in range(5)]
    graph = SparseChunkDependencyGraph(
        neighbor_indices=torch.tensor([[-1], [2], [0], [1], [1]]),
        edge_weights=torch.tensor([[0.0], [1.0], [1.0], [1.0], [1.0]]),
    )

    attention_only = DualSignalScorer(alpha=1.0, beta=0.0, window_size=2)
    routing_only = DualSignalScorer(alpha=0.0, beta=1.0, window_size=2)
    direct_scores = attention_only.forward(attention_obs, chunks)
    routed_scores = routing_only.forward(
        attention_obs, chunks, dependency_graph=graph
    )

    assert direct_scores[2] < direct_scores[1]
    assert routed_scores[2] > routed_scores[1]


def test_scorer_details_expose_independent_tier_and_eviction_scores():
    attention_obs = torch.tensor(
        [[[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]]], dtype=torch.float32
    )
    chunks = [torch.tensor([i]) for i in range(3)]
    graph = SparseChunkDependencyGraph(
        neighbor_indices=torch.tensor([[-1], [2], [0]], dtype=torch.long),
        edge_weights=torch.tensor([[0.0], [1.0], [1.0]]),
    )
    scorer = DualSignalScorer(alpha=0.6, beta=0.4, window_size=2)

    result = scorer.forward_with_details(
        attention_obs, chunks, dependency_graph=graph
    )

    assert result.attention_scores.shape == (3,)
    assert result.dependency_scores.shape == (3,)
    assert torch.allclose(
        result.chunk_scores,
        0.6 * result.attention_scores + 0.4 * result.dependency_scores,
    )


def test_hf_attention_adapter_builds_sparse_graph_without_self_loops():
    attention = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    attention[0, 0, 0, 0] = 1.0
    attention[0, 0, 1, 0] = 0.8
    attention[0, 0, 1, 1] = 0.2
    attention[0, 0, 2:, 0:2] = 0.1
    attention[0, 0, 2:, 2:] = 0.4
    chunk_map = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    graph = build_sparse_chunk_dependency_graph(
        [attention], chunk_map=chunk_map, top_k=2
    )

    assert graph.num_chunks == 3
    assert graph.top_k == 2
    for chunk_id, neighbors in enumerate(graph.neighbor_indices.tolist()):
        assert chunk_id not in neighbors
    assert graph.neighbor_indices[1, 0].item() == 0


def test_attention_row_aggregation_preserves_layer_modes():
    first = torch.ones(1, 2, 3, 4)
    second = torch.full((1, 2, 3, 4), 4.0)

    last_rows = aggregate_attention_rows([first, second], mode="last")
    all_rows = aggregate_attention_rows([first, second], mode="all")
    uniform_rows = aggregate_attention_rows(
        [first, second], mode="all", layer_weighting="uniform"
    )

    assert torch.allclose(last_rows, torch.full((3, 4), 4.0))
    assert torch.allclose(all_rows, torch.full((3, 4), 3.0))
    assert torch.allclose(uniform_rows, torch.full((3, 4), 2.5))


def test_reverse_routing_rescues_historical_bridge_chunk():
    graph = SparseChunkDependencyGraph(
        neighbor_indices=torch.tensor([[-1], [0], [1]], dtype=torch.long),
        edge_weights=torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32),
    )

    # The final query directly values chunk 2. Its causal edge to chunk 1
    # should protect that older bridge; relevance must not flow forward to 2.
    routed = graph.route(torch.tensor([0.0, 0.0, 1.0]))

    assert torch.allclose(routed, torch.tensor([0.0, 1.0, 1.0]))


def test_all_layer_sparse_graph_matches_dense_mathematical_oracle():
    torch.manual_seed(23)
    sequence_length = 6
    chunk_map = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    causal_mask = torch.tril(torch.ones(sequence_length, sequence_length)).bool()
    attentions = []
    for _layer in range(3):
        logits = torch.randn(1, 2, sequence_length, sequence_length)
        logits = logits.masked_fill(~causal_mask, float("-inf"))
        attentions.append(torch.softmax(logits, dim=-1))

    graph = build_sparse_chunk_dependency_graph(
        attentions,
        chunk_map=chunk_map,
        top_k=2,
        mode="all",
        layer_weighting="linear",
        offload_to_cpu=False,
    )

    layer_weights = torch.tensor([1.0, 2.0, 3.0]) / 6.0
    dense_rows = sum(
        layer_weights[layer] * attentions[layer][0].mean(dim=0)
        for layer in range(3)
    )
    expected_indices = torch.full((3, 2), -1, dtype=torch.long)
    expected_weights = torch.zeros((3, 2), dtype=torch.float32)
    for query_chunk in range(3):
        query_rows = dense_rows[chunk_map == query_chunk]
        token_mass = query_rows.sum(dim=0)
        chunk_mass = torch.zeros(3)
        chunk_mass.scatter_add_(0, chunk_map, token_mass)
        chunk_mass /= float(query_rows.shape[0])
        chunk_mass[query_chunk] = 0.0
        edge_count = min(2, int((chunk_mass > 0).sum().item()))
        if edge_count:
            values, indices = torch.topk(chunk_mass, edge_count)
            expected_indices[query_chunk, :edge_count] = indices
            expected_weights[query_chunk, :edge_count] = values / values.sum()

    assert torch.equal(graph.neighbor_indices, expected_indices)
    assert torch.allclose(graph.edge_weights, expected_weights, atol=1e-6, rtol=1e-6)
