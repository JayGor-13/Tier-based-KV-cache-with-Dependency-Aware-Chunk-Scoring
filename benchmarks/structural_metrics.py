"""Evidence-retention and attention-structure metrics for paper runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class EvidenceTargets:
    evidence_texts: tuple[str, ...]
    matched_texts: tuple[str, ...]
    token_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: list(value) for key, value in payload.items()}


def _coerce_sentences(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _hotpot_context_map(context: Any) -> dict[str, list[str]]:
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("text") or []
        return {
            str(title): _coerce_sentences(
                sentences[index] if index < len(sentences) else []
            )
            for index, title in enumerate(titles)
        }
    result: dict[str, list[str]] = {}
    if isinstance(context, (list, tuple)):
        for index, item in enumerate(context):
            if isinstance(item, dict):
                title = str(item.get("title", f"Document {index + 1}"))
                result[title] = _coerce_sentences(
                    item.get("sentences", item.get("text", []))
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                result[str(item[0])] = _coerce_sentences(item[1])
    return result


def _hotpot_support_pairs(supporting_facts: Any) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or supporting_facts.get("titles") or []
        ids = (
            supporting_facts.get("sent_id")
            or supporting_facts.get("sent_ids")
            or supporting_facts.get("sentence_id")
            or []
        )
        for index, title in enumerate(titles):
            if index < len(ids):
                pairs.append((str(title), int(ids[index])))
    elif isinstance(supporting_facts, (list, tuple)):
        for item in supporting_facts:
            if isinstance(item, dict):
                title = item.get("title", item.get("document"))
                sentence_id = item.get("sent_id", item.get("sentence_id"))
                if title is not None and sentence_id is not None:
                    pairs.append((str(title), int(sentence_id)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), int(item[1])))
    return pairs


def evidence_texts_from_record(
    record: dict[str, Any],
    *,
    adapter: str | None,
    gold: str | None,
) -> list[str]:
    """Extract answer-critical text that is expected to occur in the prompt."""
    adapter = str(adapter or "").lower()
    texts: list[str] = []
    if adapter == "niah":
        needle = str(record.get("needle") or gold or "").strip()
        if needle:
            texts.extend((f"The secret retrieval key is {needle}.", needle))
    elif adapter == "hotpotqa":
        context = _hotpot_context_map(record.get("context"))
        for title, sentence_id in _hotpot_support_pairs(
            record.get("supporting_facts")
        ):
            sentences = context.get(title, [])
            if 0 <= sentence_id < len(sentences):
                sentence = sentences[sentence_id].strip()
                if sentence:
                    texts.append(sentence)
        if not texts and gold:
            texts.append(str(gold).strip())

    # Preserve order while removing empty and duplicate strings.
    return list(dict.fromkeys(text for text in texts if text))


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().flatten().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _subsequence_starts(sequence: Sequence[int], pattern: Sequence[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    width = len(pattern)
    return [
        start
        for start in range(len(sequence) - width + 1)
        if list(sequence[start : start + width]) == list(pattern)
    ]


def locate_evidence_targets(
    *,
    tokenizer: Any,
    input_ids: torch.Tensor | Sequence[int],
    evidence_texts: Iterable[str],
) -> EvidenceTargets:
    """Locate every exact tokenized evidence span in the prepared model input."""
    sequence = (
        [int(value) for value in input_ids.detach().cpu().flatten().tolist()]
        if isinstance(input_ids, torch.Tensor)
        else [int(value) for value in input_ids]
    )
    requested = tuple(dict.fromkeys(str(text).strip() for text in evidence_texts if str(text).strip()))
    matched: list[str] = []
    positions: set[int] = set()
    for text in requested:
        variants = (text, " " + text, "\n" + text)
        found_for_text: set[int] = set()
        for variant in variants:
            pattern = _encode_without_special_tokens(tokenizer, variant)
            for start in _subsequence_starts(sequence, pattern):
                found_for_text.update(range(start, start + len(pattern)))
        if found_for_text:
            matched.append(text)
            positions.update(found_for_text)
    return EvidenceTargets(
        evidence_texts=requested,
        matched_texts=tuple(matched),
        token_indices=tuple(sorted(positions)),
    )


def compute_evidence_retention(
    targets: EvidenceTargets,
    *,
    kept_indices: torch.Tensor | Sequence[int],
    chunk_map: torch.Tensor | Sequence[int] | None = None,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Compute evidence survival under the repository's global cache mask.

    This is an auditable evidence-token eviction proxy.  It is deliberately not
    described as a head/layer reachability GER because the current policy uses
    one shared position mask across every layer and KV head.
    """
    kept = set(
        int(value)
        for value in (
            kept_indices.detach().cpu().flatten().tolist()
            if isinstance(kept_indices, torch.Tensor)
            else kept_indices
        )
    )
    critical = set(targets.token_indices)
    retained = critical & kept
    evicted = critical - kept
    total = len(critical)
    evidence_count = len(targets.evidence_texts)
    matched_count = len(targets.matched_texts)

    chunk_ids: set[int] = set()
    retained_chunk_ids: set[int] = set()
    if chunk_map is not None and critical:
        mapping = (
            chunk_map.detach().cpu().flatten().tolist()
            if isinstance(chunk_map, torch.Tensor)
            else list(chunk_map)
        )
        for token_index in critical:
            if 0 <= token_index < len(mapping):
                chunk_id = int(mapping[token_index])
                chunk_ids.add(chunk_id)
                if token_index in retained:
                    retained_chunk_ids.add(chunk_id)

    evidence_eviction_ratio = len(evicted) / float(total) if total else None
    return {
        "evidence_text_count": evidence_count,
        "matched_evidence_text_count": matched_count,
        "evidence_localization_rate": (
            matched_count / float(evidence_count) if evidence_count else None
        ),
        "critical_token_count": total,
        "retained_critical_tokens": len(retained),
        "evicted_critical_tokens": len(evicted),
        "evidence_token_retention": len(retained) / float(total) if total else None,
        "evidence_token_eviction_ratio": evidence_eviction_ratio,
        "global_eviction_ratio": evidence_eviction_ratio,
        "global_eviction_ratio_definition": "global_mask_evidence_token_proxy",
        "evidence_depth_min": (
            min(critical) / float(max(1, int(sequence_length) - 1))
            if critical and sequence_length is not None
            else None
        ),
        "evidence_depth_max": (
            max(critical) / float(max(1, int(sequence_length) - 1))
            if critical and sequence_length is not None
            else None
        ),
        "evidence_chunk_count": len(chunk_ids),
        "retained_evidence_chunks": len(retained_chunk_ids),
        "evidence_chunk_survival": (
            len(retained_chunk_ids) / float(len(chunk_ids)) if chunk_ids else None
        ),
    }


def compute_head_consensus(
    attention_obs: torch.Tensor,
    *,
    top_k: int = 16,
) -> dict[str, float | int | None]:
    """Measure pairwise overlap of head-level top-attended key positions."""
    attention = attention_obs.detach().to(dtype=torch.float32, device="cpu")
    if attention.ndim == 3:
        attention = attention.unsqueeze(0)
    if attention.ndim != 4 or attention.shape[1] < 2:
        return {
            "head_consensus": None,
            "head_diversity": None,
            "head_pairs": 0,
            "head_consensus_top_k": int(top_k),
        }

    similarities: list[float] = []
    key_count = int(attention.shape[-1])
    width = min(max(1, int(top_k)), key_count)
    for layer in attention:
        head_scores = layer.sum(dim=1)
        head_sets = [
            set(torch.topk(scores, k=width).indices.tolist())
            for scores in head_scores
        ]
        for left, right in combinations(head_sets, 2):
            union = left | right
            similarities.append(len(left & right) / float(len(union)) if union else 1.0)
    consensus = sum(similarities) / len(similarities) if similarities else None
    return {
        "head_consensus": consensus,
        "head_diversity": 1.0 - consensus if consensus is not None else None,
        "head_pairs": len(similarities),
        "head_consensus_top_k": width,
    }


__all__ = [
    "EvidenceTargets",
    "compute_evidence_retention",
    "compute_head_consensus",
    "evidence_texts_from_record",
    "locate_evidence_targets",
]
