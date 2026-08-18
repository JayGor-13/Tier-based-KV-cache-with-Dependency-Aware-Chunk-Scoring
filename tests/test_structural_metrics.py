import torch

from benchmarks.structural_metrics import (
    compute_evidence_retention,
    compute_head_consensus,
    evidence_texts_from_record,
    locate_evidence_targets,
)


class _WhitespaceTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, **_kwargs):
        ids = []
        for word in str(text).split():
            normalized = word.strip(".,")
            self.vocab.setdefault(normalized, len(self.vocab) + 1)
            ids.append(self.vocab[normalized])
        return {"input_ids": ids}


class _TrailingBoundaryTokenizer:
    """Fixture whose evidence token changes when followed by whitespace."""

    def __call__(self, text, **_kwargs):
        values = {
            "prefix Evidence suffix": [10, 20, 30],
            "Evidence": [21],
            " Evidence": [22],
            "\nEvidence": [23],
            "Evidence ": [20],
        }
        return {"input_ids": values.get(str(text), [99])}


def test_niah_evidence_token_eviction_is_auditable():
    tokenizer = _WhitespaceTokenizer()
    prompt = "archive The secret retrieval key is KEY-7. record"
    input_ids = tokenizer(prompt)["input_ids"]
    texts = evidence_texts_from_record(
        {"needle": "KEY-7"}, adapter="niah", gold="KEY-7"
    )
    targets = locate_evidence_targets(
        tokenizer=tokenizer,
        input_ids=input_ids,
        evidence_texts=texts,
    )
    metrics = compute_evidence_retention(
        targets,
        kept_indices=[0, 1, 2, 3, 5, 6, 7],
        chunk_map=[0, 1, 1, 1, 1, 2, 2, 3],
        sequence_length=8,
    )
    assert metrics["critical_token_count"] == 6
    assert metrics["evicted_critical_tokens"] == 1
    assert metrics["evidence_token_eviction_ratio"] == 1 / 6
    assert metrics["evidence_chunk_survival"] == 1.0


def test_head_consensus_reports_identical_heads_as_one():
    attention = torch.tensor(
        [[[[0.8, 0.2]], [[0.8, 0.2]]]], dtype=torch.float32
    )
    result = compute_head_consensus(attention, top_k=1)
    assert result["head_consensus"] == 1.0
    assert result["head_diversity"] == 0.0


def test_evidence_locator_covers_context_sensitive_trailing_boundary():
    targets = locate_evidence_targets(
        tokenizer=_TrailingBoundaryTokenizer(),
        input_ids=[10, 20, 30],
        evidence_texts=["Evidence"],
    )

    assert targets.matched_texts == ("Evidence",)
    assert targets.token_indices == (1,)
