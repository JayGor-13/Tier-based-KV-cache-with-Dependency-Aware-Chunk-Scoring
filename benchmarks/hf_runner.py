"""HuggingFace-backed experiment runner for TDC-KV."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import torch

from benchmarks.eval_metrics import (
    CacheMetrics,
    aggregate_grouped_runs,
    compute_cache_metrics,
    judge_gsm8k_prediction,
    summarize_generation_parity,
    summarize_cache_metrics,
    summarize_qa,
    validate_budget_contract,
)
from benchmarks.gsm8k_protocol import (
    CHUNKKV_GSM8K_8SHOT_PROTOCOL,
    DEFAULT_PROTOCOL,
    build_chunkkv_gsm8k_prompt,
    canonicalize_protocol,
    protocol_metadata,
    validate_protocol_for_adapter,
)
from src.baselines.chunkkv import evict_chunkkv
from src.baselines.h2o import evict_h2o
from src.baselines.snapkv import evict_snapkv
from src.baselines.streamingllm import evict_streamingllm
from src.core.chunker import SentenceBoundaryChunkConstructor
from src.core.evictor import EvictionResult, evict_kv_cache
from src.core.masker import MaskerResult, assign_protection_tiers
from src.core.scorer import DualSignalScorer
from src.models.cache_utils import (
    generate_text,
    generate_text_with_evicted_cache,
    load_hf_model_and_tokenizer,
    prepare_prompt,
    run_hf_prefill,
)


SUPPORTED_METHODS = {
    "fullkv",
    "streamingllm",
    "h2o",
    "snapkv",
    "chunkkv",
    "tdc_kv",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset source and field mapping."""

    name: str
    source: str
    split: str | None = None
    config: str | None = None
    prompt_field: str = "prompt"
    answer_field: str | None = None
    id_field: str | None = None
    prompt_template: str | None = None
    adapter: str | None = None
    protocol: str = DEFAULT_PROTOCOL
    options: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_key_value_spec(text: str) -> dict[str, str]:
    """Parse `key=value,key=value` CLI strings."""
    parsed: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Dataset spec part `{part}` must be key=value.")
        key, value = part.split("=", 1)
        parsed[key.strip().lower().replace("-", "_")] = value.strip()
    return parsed


def parse_dataset_spec(text: str) -> DatasetSpec:
    """Parse a dataset spec used by `scripts/run_hf_grid.py`."""
    raw = parse_key_value_spec(text)
    source = raw.get("source") or raw.get("path") or raw.get("dataset")
    if not source:
        raise ValueError("Dataset spec requires `source=...`.")

    adapter = _canonical_dataset_adapter(raw.get("adapter") or raw.get("task") or source)
    source = _canonical_dataset_source(source, adapter)
    defaults = _dataset_defaults(adapter)
    source_path = Path(source)
    default_name = (
        adapter
        or (source_path.stem if source_path.suffix else source.replace("/", "_"))
    )
    reserved = {
        "adapter",
        "answer",
        "answer_field",
        "config",
        "dataset",
        "id",
        "id_field",
        "name",
        "path",
        "prompt",
        "prompt_field",
        "protocol",
        "source",
        "split",
        "task",
        "template",
    }
    protocol = canonicalize_protocol(raw.get("protocol"))
    validate_protocol_for_adapter(protocol, adapter)
    if protocol != DEFAULT_PROTOCOL and raw.get("template"):
        raise ValueError(
            "Paper evaluation protocols cannot be combined with a custom template."
        )
    return DatasetSpec(
        name=raw.get("name", default_name),
        source=source,
        split=raw.get("split", defaults.get("split")),
        config=raw.get("config", defaults.get("config")),
        prompt_field=raw.get("prompt_field", raw.get("prompt", defaults["prompt_field"])),
        answer_field=raw.get("answer_field", raw.get("answer", defaults["answer_field"])),
        id_field=raw.get("id_field", raw.get("id", defaults.get("id_field"))),
        prompt_template=raw.get("template"),
        adapter=adapter,
        protocol=protocol,
        options={key: value for key, value in raw.items() if key not in reserved},
    )


def _canonical_dataset_adapter(text: str | None) -> str | None:
    if not text:
        return None
    normalized = text.strip().lower().replace("-", "_")
    aliases = {
        "gsm8k": "gsm8k",
        "openai/gsm8k": "gsm8k",
        "openai_gsm8k": "gsm8k",
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "hotpot_qa": "hotpotqa",
        "hotpotqa/hotpot_qa": "hotpotqa",
        "hotpotqa_hotpot_qa": "hotpotqa",
        "niah": "niah",
        "needle": "niah",
        "needle_in_a_haystack": "niah",
    }
    return aliases.get(normalized)


def _canonical_dataset_source(source: str, adapter: str | None) -> str:
    if adapter == "gsm8k" and source.strip().lower().replace("-", "_") == "gsm8k":
        return "openai/gsm8k"
    if adapter == "hotpotqa" and source.strip().lower().replace("-", "_") in {
        "hotpot",
        "hotpotqa",
        "hotpot_qa",
    }:
        return "hotpotqa/hotpot_qa"
    return source


def _dataset_defaults(adapter: str | None) -> dict[str, str | None]:
    if adapter == "gsm8k":
        return {
            "split": "test",
            "config": "main",
            "prompt_field": "question",
            "answer_field": "answer",
            "id_field": None,
        }
    if adapter == "hotpotqa":
        return {
            "split": "validation",
            "config": "distractor",
            "prompt_field": "question",
            "answer_field": "answer",
            "id_field": "id",
        }
    if adapter == "niah":
        return {
            "split": None,
            "config": None,
            "prompt_field": "question",
            "answer_field": "answer",
            "id_field": "id",
        }
    return {
        "split": None,
        "config": None,
        "prompt_field": "prompt",
        "answer_field": None,
        "id_field": None,
    }


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return list(data)
        if isinstance(data, dict):
            for key in ("samples", "data", "records", "examples"):
                if isinstance(data.get(key), list):
                    return list(data[key])
            return [data]
    raise ValueError(f"Unsupported local dataset format `{path.suffix}`.")


def load_dataset_records(
    spec: DatasetSpec,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load local JSON/JSONL or a HuggingFace `datasets` source."""
    path = Path(spec.source)
    if path.exists():
        records = _load_json_records(path)
    elif spec.adapter == "niah" or spec.source.strip().lower() == "niah":
        records = _generate_niah_records(spec, max_samples=max_samples)
    else:
        dataset_source = {
            "gsm8k": "openai/gsm8k",
            "hotpot_qa": "hotpotqa/hotpot_qa",
            "hotpotqa": "hotpotqa/hotpot_qa",
        }.get(spec.source, spec.source)
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise ModuleNotFoundError(
                "HuggingFace dataset loading requires `datasets`. Install "
                "dependencies from requirements.txt or environment.yml."
            ) from exc

        split = spec.split or "validation"
        if spec.config:
            dataset = load_dataset(dataset_source, spec.config, split=split)
        else:
            dataset = load_dataset(dataset_source, split=split)
        records = [dict(row) for row in dataset]

    if max_samples is not None:
        records = records[: max(0, int(max_samples))]
    return records


def _get_nested(record: dict[str, Any], field: str | None) -> Any:
    if not field:
        return None
    if field in record:
        return record[field]

    current: Any = record
    for part in field.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _answer_to_text(answer: Any) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, dict):
        for key in ("text", "answer", "answers", "value"):
            if key in answer:
                return _answer_to_text(answer[key])
        return json.dumps(answer, ensure_ascii=False)
    if isinstance(answer, list):
        if not answer:
            return ""
        return _answer_to_text(answer[0])
    return str(answer)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _record_format_values(
    record: dict[str, Any],
    spec: DatasetSpec,
    prompt_value: Any,
    gold: str | None,
) -> _SafeFormatDict:
    values = _SafeFormatDict(
        {
            key: value
            for key, value in record.items()
            if isinstance(value, (str, int, float, bool))
        }
    )
    values["prompt"] = str(prompt_value)
    values["answer"] = gold or ""
    values["gold"] = gold or ""
    if spec.adapter == "hotpotqa":
        values["context"] = _format_hotpot_context(record.get("context"))
        values["supporting_facts"] = _format_hotpot_supporting_facts(
            record.get("supporting_facts")
        )
    elif spec.adapter == "niah":
        values["context"] = str(record.get("context", ""))
        values["needle"] = str(record.get("needle", gold or ""))
    return values


def _format_with_template(
    template: str,
    values: dict[str, Any],
) -> str:
    return template.format_map(values)


def _build_gsm8k_prompt(values: dict[str, Any]) -> str:
    return _format_with_template(
        (
            "Answer the grade-school math problem. Reason step by step, then "
            "write the final answer on a new line as #### <number>.\n\n"
            "Question: {prompt}\n"
            "Answer:"
        ),
        values,
    )


def _coerce_sentences(sentences: Any) -> list[str]:
    if sentences is None:
        return []
    if isinstance(sentences, str):
        return [sentences]
    if isinstance(sentences, list):
        return [str(sentence) for sentence in sentences]
    return [str(sentences)]


def _iter_hotpot_context(context: Any) -> list[tuple[str, list[str]]]:
    if context is None:
        return []
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("text") or []
        return [
            (str(title), _coerce_sentences(sentences[index] if index < len(sentences) else []))
            for index, title in enumerate(titles)
        ]
    if isinstance(context, list):
        paragraphs: list[tuple[str, list[str]]] = []
        for index, item in enumerate(context):
            if isinstance(item, dict):
                title = item.get("title", f"Document {index + 1}")
                sentences = item.get("sentences", item.get("text", []))
                paragraphs.append((str(title), _coerce_sentences(sentences)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                paragraphs.append((str(item[0]), _coerce_sentences(item[1])))
            else:
                paragraphs.append((f"Document {index + 1}", _coerce_sentences(item)))
        return paragraphs
    return [("Document 1", [str(context)])]


def _format_hotpot_context(context: Any) -> str:
    paragraphs = _iter_hotpot_context(context)
    if not paragraphs:
        return ""
    formatted = []
    for index, (title, sentences) in enumerate(paragraphs, start=1):
        body = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
        formatted.append(f"[{index}] {title}\n{body}".rstrip())
    return "\n\n".join(formatted)


def _format_hotpot_supporting_facts(supporting_facts: Any) -> str:
    if supporting_facts is None:
        return ""
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or supporting_facts.get("titles") or []
        sentence_ids = (
            supporting_facts.get("sent_id")
            or supporting_facts.get("sent_ids")
            or supporting_facts.get("sentence_id")
            or []
        )
        facts = []
        for index, title in enumerate(titles):
            sentence_id = sentence_ids[index] if index < len(sentence_ids) else "?"
            facts.append(f"{title}:{sentence_id}")
        return "; ".join(facts)
    if isinstance(supporting_facts, list):
        facts = []
        for fact in supporting_facts:
            if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                facts.append(f"{fact[0]}:{fact[1]}")
            elif isinstance(fact, dict):
                title = fact.get("title", fact.get("document", "?"))
                sent_id = fact.get("sent_id", fact.get("sentence_id", "?"))
                facts.append(f"{title}:{sent_id}")
            else:
                facts.append(str(fact))
        return "; ".join(facts)
    return str(supporting_facts)


def _build_hotpotqa_prompt(values: dict[str, Any]) -> str:
    return _format_with_template(
        (
            "Answer the question using only the provided Wikipedia context. "
            "Give a concise answer.\n\n"
            "Context:\n{context}\n\n"
            "Question: {prompt}\n"
            "Answer:"
        ),
        values,
    )


def _build_niah_prompt(values: dict[str, Any]) -> str:
    return _format_with_template(
        (
            "You are given a long context. Retrieve the secret key exactly.\n\n"
            "Context:\n{context}\n\n"
            "Question: {prompt}\n"
            "Answer:"
        ),
        values,
    )


def build_prompt_from_record(
    record: dict[str, Any],
    spec: DatasetSpec,
    *,
    prompt_template: str | None = None,
) -> tuple[str, str | None]:
    """Return formatted prompt and optional gold answer text."""
    prompt_value = _get_nested(record, spec.prompt_field)
    if prompt_value is None:
        raise KeyError(
            f"Prompt field `{spec.prompt_field}` not found for dataset `{spec.name}`."
        )

    gold = _answer_to_text(_get_nested(record, spec.answer_field))
    values = _record_format_values(record, spec, prompt_value, gold)

    protocol = canonicalize_protocol(spec.protocol)
    validate_protocol_for_adapter(protocol, spec.adapter)
    template = prompt_template or spec.prompt_template
    if protocol != DEFAULT_PROTOCOL and template:
        raise ValueError(
            "Paper evaluation protocols cannot be combined with a custom template."
        )
    if template:
        return _format_with_template(template, values), gold
    if protocol == CHUNKKV_GSM8K_8SHOT_PROTOCOL:
        return build_chunkkv_gsm8k_prompt(str(prompt_value)), gold
    if spec.adapter == "gsm8k":
        return _build_gsm8k_prompt(values), gold
    if spec.adapter == "hotpotqa":
        return _build_hotpotqa_prompt(values), gold
    if spec.adapter == "niah":
        return _build_niah_prompt(values), gold
    return _format_with_template("{prompt}", values), gold


_NIAH_FILLER_WORDS = (
    "archive",
    "buffer",
    "catalog",
    "detail",
    "entry",
    "figure",
    "ledger",
    "marker",
    "note",
    "object",
    "passage",
    "record",
    "section",
    "thread",
    "value",
    "window",
)


def _parse_int_option(options: dict[str, str], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key in options:
            return int(options[key])
    return int(default)


def _parse_float_option(
    options: dict[str, str], keys: tuple[str, ...], default: float
) -> float:
    for key in keys:
        if key in options:
            return float(options[key])
    return float(default)


def _generate_filler_words(count: int, *, offset: int) -> list[str]:
    words = []
    for idx in range(max(0, int(count))):
        word = _NIAH_FILLER_WORDS[(idx + offset) % len(_NIAH_FILLER_WORDS)]
        words.append(f"{word}{(idx + offset) % 997}")
    return words


def _generate_niah_records(
    spec: DatasetSpec,
    *,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    options = spec.options
    count = max_samples if max_samples is not None else _parse_int_option(
        options, ("num_samples", "samples"), 1
    )
    count = max(0, int(count))
    context_tokens = _parse_int_option(
        options,
        ("context_length", "context_tokens", "target_context_tokens"),
        4096,
    )
    needle_depth = _parse_float_option(
        options, ("needle_depth", "depth", "needle_position"), 0.5
    )
    if not math.isfinite(needle_depth) or needle_depth < 0.0 or needle_depth > 1.0:
        raise ValueError("NIAH needle_depth must be a finite value in [0, 1].")
    seed = _parse_int_option(options, ("seed",), 13)
    needle_prefix = options.get("needle_prefix", "NIAH")
    question = options.get("question", "What is the secret retrieval key?")

    records = []
    needle_template = "The secret retrieval key is {needle}."
    for sample_index in range(count):
        needle = f"{needle_prefix}-{seed + sample_index:06d}"
        needle_words = needle_template.format(needle=needle).split()
        filler_count = max(0, int(context_tokens) - len(needle_words))
        filler_words = _generate_filler_words(
            filler_count,
            offset=seed + sample_index * 31,
        )
        insert_at = min(
            len(filler_words),
            max(0, int(round(needle_depth * len(filler_words)))),
        )
        words = (
            filler_words[:insert_at]
            + needle_words
            + filler_words[insert_at:]
        )
        context = " ".join(words)
        records.append(
            {
                "id": f"niah_{context_tokens}_{needle_depth:g}_{sample_index}",
                "context": context,
                "question": question,
                "answer": needle,
                "needle": needle,
                "needle_depth": needle_depth,
                "target_context_tokens": int(context_tokens),
                "needle_word_index": insert_at,
            }
        )
    return records


def resolve_budgets(
    sequence_length: int,
    *,
    budgets: Iterable[int],
    budget_ratios: Iterable[float],
) -> list[int]:
    """Resolve absolute and ratio budgets for one tokenized sample."""
    return [
        int(configuration["budget"])
        for configuration in resolve_budget_configurations(
            sequence_length,
            budgets=budgets,
            budget_ratios=budget_ratios,
        )
    ]


def resolve_budget_configurations(
    sequence_length: int,
    *,
    budgets: Iterable[int],
    budget_ratios: Iterable[float],
) -> list[dict[str, Any]]:
    """Resolve token budgets while retaining their experiment-level origin."""
    specifications_by_budget: dict[int, list[dict[str, Any]]] = {}

    for requested_budget in sorted({int(value) for value in budgets if int(value) >= 0}):
        resolved = max(0, min(requested_budget, int(sequence_length)))
        specifications_by_budget.setdefault(resolved, []).append(
            {"type": "absolute", "value": requested_budget}
        )

    for ratio in sorted({float(value) for value in budget_ratios}):
        resolved = max(
            1,
            min(int(round(ratio * sequence_length)), int(sequence_length)),
        )
        specifications_by_budget.setdefault(resolved, []).append(
            {"type": "ratio", "value": ratio}
        )

    configurations = []
    for resolved_budget, specifications in sorted(specifications_by_budget.items()):
        if len(specifications) == 1:
            descriptor = specifications[0]
        else:
            descriptor = {"type": "combined", "value": specifications}
        configurations.append(
            {
                "budget": int(resolved_budget),
                "descriptor": descriptor,
                "specifications": specifications,
            }
        )
    return configurations


def _slice_attention_window(attention_obs: torch.Tensor, window_size: int) -> torch.Tensor:
    actual = min(max(1, int(window_size)), int(attention_obs.shape[-1]))
    return attention_obs[..., -actual:, :]


def _tier_counts(tiers: torch.Tensor) -> dict[str, int]:
    return {
        "tier0": int((tiers == 0).sum().item()),
        "tier1": int((tiers == 1).sum().item()),
        "tier2": int((tiers == 2).sum().item()),
    }


def _chunk_size_metrics(chunks: list[torch.Tensor]) -> dict[str, int | float]:
    sizes = [int(chunk.numel()) for chunk in chunks]
    if not sizes:
        return {
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "avg_chunk_size": 0.0,
        }
    return {
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "avg_chunk_size": sum(sizes) / float(len(sizes)),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sha256_token_ids(token_ids: Iterable[int]) -> str:
    serialized = json.dumps(
        [int(token_id) for token_id in token_ids],
        separators=(",", ":"),
    )
    return _sha256_text(serialized)


def _model_revision(bundle: Any) -> str | None:
    candidates = [
        getattr(getattr(bundle.model, "config", None), "_commit_hash", None),
        getattr(bundle.tokenizer, "init_kwargs", {}).get("_commit_hash"),
    ]
    return next((str(value) for value in candidates if value), None)


def _judge_prediction(
    spec: DatasetSpec,
    prediction: str,
    gold: str | None,
) -> dict | None:
    if gold is None:
        return None
    if spec.adapter == "gsm8k":
        return judge_gsm8k_prediction(
            prediction,
            gold,
            protocol=canonicalize_protocol(spec.protocol),
        )
    return None


def normalize_methods(methods: Iterable[str] | None) -> list[str]:
    """Normalize and validate experiment method names."""
    if methods is None:
        return ["tdc_kv"]

    aliases = {
        "tdc-kv": "tdc_kv",
        "tdckv": "tdc_kv",
        "full": "fullkv",
        "full_kv": "fullkv",
        "streaming": "streamingllm",
        "streaming_llm": "streamingllm",
        "skv": "snapkv",
    }
    normalized: list[str] = []
    for raw in methods:
        method = aliases.get(raw.strip().lower(), raw.strip().lower())
        if not method:
            continue
        if method not in SUPPORTED_METHODS:
            supported = ", ".join(sorted(SUPPORTED_METHODS))
            raise ValueError(f"Unsupported method `{raw}`. Supported methods: {supported}.")
        if method not in normalized:
            normalized.append(method)

    if not normalized:
        raise ValueError("At least one method is required.")
    return normalized


def _baseline_attention(attention_obs: torch.Tensor) -> torch.Tensor:
    """Return a 3D attention tensor for token-level baselines."""
    if attention_obs.ndim == 3:
        return attention_obs
    if attention_obs.ndim == 4:
        return attention_obs[-1]
    raise ValueError(
        f"Expected attention shape [H,w,t] or [L,H,w,t], got {tuple(attention_obs.shape)}."
    )


def _run_eviction_method(
    method: str,
    prefill: Any,
    *,
    budget: int,
    theta: float,
    recent_window: int,
    chunk_scores: torch.Tensor,
    protection_scores: torch.Tensor,
    tier1_score_mode: str,
    attention_obs: torch.Tensor,
    allow_level2_fallback: bool,
) -> tuple[EvictionResult, MaskerResult | None]:
    """Apply one compressed-cache method to an HF prefill record."""
    method = normalize_methods([method])[0]
    if method == "fullkv":
        raise ValueError("`fullkv` is handled before eviction.")

    if method == "tdc_kv":
        if tier1_score_mode == "dependency":
            tier1_theta = float(theta)
            tier1_scores = protection_scores
        elif tier1_score_mode == "fused":
            tier1_theta = float(theta)
            tier1_scores = None
        elif tier1_score_mode == "none":
            tier1_theta = 0.0
            tier1_scores = None
        else:
            raise ValueError(
                "tier1_score_mode must be `dependency`, `fused`, or `none`."
            )
        masker_result = assign_protection_tiers(
            chunk_scores=chunk_scores,
            chunks=prefill.chunks,
            theta=tier1_theta,
            recent_window=int(recent_window),
            sequence_length=prefill.sequence_length,
            protection_scores=tier1_scores,
            return_details=True,
        )
        eviction = evict_kv_cache(
            mask_tiers=masker_result.tiers,
            chunk_scores=chunk_scores,
            chunks=prefill.chunks,
            k_cache=prefill.k_cache,
            v_cache=prefill.v_cache,
            budget=int(budget),
            allow_level2_fallback=allow_level2_fallback,
        )
        return eviction, masker_result

    if method == "streamingllm":
        return (
            evict_streamingllm(
                k_cache=prefill.k_cache,
                v_cache=prefill.v_cache,
                budget=int(budget),
            ),
            None,
        )

    if method == "chunkkv":
        return (
            evict_chunkkv(
                chunk_scores=chunk_scores,
                chunks=prefill.chunks,
                k_cache=prefill.k_cache,
                v_cache=prefill.v_cache,
                budget=int(budget),
                theta=float(theta),
                recent_window=int(recent_window),
                sequence_length=prefill.sequence_length,
            ),
            None,
        )

    baseline_attention = _baseline_attention(attention_obs)
    if method == "snapkv":
        return (
            evict_snapkv(
                attention_obs=baseline_attention,
                k_cache=prefill.k_cache,
                v_cache=prefill.v_cache,
                budget=int(budget),
                recent_window=int(recent_window),
            ),
            None,
        )
    if method == "h2o":
        return (
            evict_h2o(
                attention_obs=baseline_attention,
                k_cache=prefill.k_cache,
                v_cache=prefill.v_cache,
                budget=int(budget),
                recent_window=int(recent_window),
            ),
            None,
        )

    raise ValueError(f"Unsupported method `{method}`.")


def _append_method_quality(
    method_qa_rows: dict[str, list[dict[str, str]]],
    method: str,
    prediction: str,
    gold: str | None,
    dataset: str | None = None,
) -> None:
    if gold is not None:
        row = {"prediction": prediction, "gold": gold}
        if dataset is not None:
            row["dataset"] = dataset
        method_qa_rows.setdefault(method, []).append(row)


def _error_row(
    *,
    model_name: str,
    dataset_name: str,
    sample_id: str,
    error: Exception,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "model": model_name,
        "dataset": dataset_name,
        "sample_id": sample_id,
        "config": config or {},
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run_hf_grid(
    *,
    model_names: list[str],
    dataset_specs: list[DatasetSpec],
    budgets: list[int],
    budget_ratios: list[float],
    thetas: list[float],
    recent_windows: list[int],
    alphas: list[float],
    methods: list[str] | None = None,
    max_samples: int | None = None,
    max_length: int | None = 2048,
    max_new_tokens: int = 0,
    prompt_template: str | None = None,
    attention_mode: str = "last",
    layer_index: int = -1,
    min_chunk_tokens: int = 5,
    max_chunk_tokens: int = 64,
    min_budget_utilization: float = 0.99,
    max_budget_shortfall_tokens: int = 1,
    dependency_top_k: int = 8,
    prefill_block_size: int = 128,
    tier1_score_mode: str = "dependency",
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = "eager",
    allow_level2_fallback: bool = True,
    continue_on_error: bool = True,
    progress: bool = False,
    run_fullkv_parity: bool = False,
    prompt_serialization: str = "auto",
) -> dict[str, Any]:
    """Run TDC-KV over all requested model/dataset/parameter combinations."""
    if not model_names:
        raise ValueError("At least one model is required.")
    if not dataset_specs:
        raise ValueError("At least one dataset is required.")
    if not recent_windows:
        raise ValueError("At least one recent window is required.")
    if not thetas:
        raise ValueError("At least one theta value is required.")
    if not alphas:
        raise ValueError("At least one alpha value is required.")
    if prefill_block_size <= 0:
        raise ValueError("prefill_block_size must be positive.")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive.")
    if max_chunk_tokens < min_chunk_tokens:
        raise ValueError(
            "max_chunk_tokens must be greater than or equal to "
            "min_chunk_tokens."
        )
    if not 0.0 <= min_budget_utilization <= 1.0:
        raise ValueError("min_budget_utilization must be in [0, 1].")
    if max_budget_shortfall_tokens < 0:
        raise ValueError("max_budget_shortfall_tokens must be non-negative.")
    if run_fullkv_parity and max_new_tokens <= 0:
        raise ValueError("FullKV parity requires max_new_tokens to be positive.")
    prompt_serialization = str(prompt_serialization).strip().lower()
    if prompt_serialization not in {"auto", "raw", "chat"}:
        raise ValueError("prompt_serialization must be `auto`, `raw`, or `chat`.")
    for spec in dataset_specs:
        protocol = canonicalize_protocol(spec.protocol)
        validate_protocol_for_adapter(protocol, spec.adapter)
        if protocol != DEFAULT_PROTOCOL and (prompt_template or spec.prompt_template):
            raise ValueError(
                "Paper evaluation protocols cannot be combined with a custom template."
            )
    experiment_methods = normalize_methods(methods)
    tier1_score_mode = tier1_score_mode.strip().lower()
    if tier1_score_mode not in {"dependency", "fused", "none"}:
        raise ValueError(
            "tier1_score_mode must be `dependency`, `fused`, or `none`."
        )
    compressed_methods = [
        method for method in experiment_methods if method != "fullkv"
    ]

    dataset_records = {
        spec.name: load_dataset_records(spec, max_samples=max_samples)
        for spec in dataset_specs
    }

    runs: list[dict[str, Any]] = []
    metric_rows: list[CacheMetrics] = []
    method_metric_rows: dict[str, list[CacheMetrics]] = {
        method: [] for method in experiment_methods
    }
    method_qa_rows: dict[str, list[dict[str, str]]] = {
        method: [] for method in experiment_methods
    }
    baseline_qa_rows: list[dict[str, str]] = []
    evicted_qa_rows: list[dict[str, str]] = []
    parity_records: list[dict[str, Any]] = []
    model_revisions: dict[str, str | None] = {}
    max_observation_window = max(int(window) for window in recent_windows)

    def report(message: str) -> None:
        if progress:
            print(message, flush=True)

    for model_name in model_names:
        report(f"[model] Loading {model_name}...")
        bundle = load_hf_model_and_tokenizer(
            model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        model_revision = _model_revision(bundle)
        model_revisions[model_name] = model_revision
        report(f"[model] Loaded {model_name}.")
        chunk_constructor = SentenceBoundaryChunkConstructor(
            tokenizer=bundle.tokenizer,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            device="cpu",
        )

        for spec in dataset_specs:
            records = dataset_records[spec.name]
            for sample_index, record in enumerate(records):
                sample_id = str(
                    _get_nested(record, spec.id_field) or f"{spec.name}_{sample_index}"
                )
                report(
                    f"[sample {sample_index + 1}/{len(records)}] "
                    f"dataset={spec.name} id={sample_id}"
                )
                try:
                    prompt, gold = build_prompt_from_record(
                        record, spec, prompt_template=prompt_template
                    )
                    prepared_prompt = prepare_prompt(
                        tokenizer=bundle.tokenizer,
                        prompt=prompt,
                        max_length=max_length,
                        serialization=prompt_serialization,
                    )
                    raw_prompt_sha256 = _sha256_text(prompt)
                    prompt_sha256 = _sha256_text(prepared_prompt.rendered_text)
                    report("  [fullkv] Generating baseline...")
                    full_generation = generate_text(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        max_length=max_length,
                        return_details=True,
                        prepared_prompt=prepared_prompt,
                    )
                    prediction = full_generation.text
                    report("  [fullkv] Baseline generation complete.")
                    if gold is not None:
                        baseline_qa_rows.append(
                            {
                                "prediction": prediction,
                                "gold": gold,
                                "dataset": spec.name,
                            }
                        )

                    report("  [prefill] Collecting blockwise attention and KV cache...")
                    prefill = run_hf_prefill(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        prompt=prompt,
                        sample_id=sample_id,
                        observation_window=max_observation_window,
                        max_length=max_length,
                        attention_mode=attention_mode,
                        layer_index=layer_index,
                        min_chunk_tokens=min_chunk_tokens,
                        max_chunk_tokens=max_chunk_tokens,
                        dependency_top_k=dependency_top_k,
                        prefill_block_size=prefill_block_size,
                        chunk_constructor=chunk_constructor,
                        prepared_prompt=prepared_prompt,
                    )
                    report(
                        f"  [prefill] Complete: tokens={prefill.sequence_length} "
                        f"blocks={prefill.prefill_blocks} chunks={len(prefill.chunks)}"
                    )
                    input_token_ids = tuple(
                        int(token_id) for token_id in prefill.input_ids.tolist()
                    )
                    prepared_input_token_ids = tuple(
                        int(token_id)
                        for token_id in prepared_prompt.input_ids[0].tolist()
                    )
                    if input_token_ids != prepared_input_token_ids:
                        raise RuntimeError(
                            "FullKV and prefill did not consume the same prepared "
                            "prompt token IDs."
                        )
                    input_token_sha256 = _sha256_token_ids(input_token_ids)
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    runs.append(
                        _error_row(
                            model_name=model_name,
                            dataset_name=spec.name,
                            sample_id=sample_id,
                            error=exc,
                        )
                    )
                    continue

                if run_fullkv_parity:
                    if prefill.next_token_id is None:
                        raise RuntimeError(
                            "FullKV parity requires a prefill next-token prediction."
                        )
                    report("  [parity] Running unpruned custom-cache generation...")
                    parity_generation = generate_text_with_evicted_cache(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        first_new_token_id=int(prefill.next_token_id),
                        max_new_tokens=max_new_tokens,
                        k_cache=prefill.k_cache,
                        v_cache=prefill.v_cache,
                        original_sequence_length=prefill.sequence_length,
                        budget=None,
                        return_details=True,
                    )
                    parity_record = {
                        "model": model_name,
                        "model_revision": model_revision,
                        "dataset": spec.name,
                        "sample_id": sample_id,
                        "protocol": canonicalize_protocol(spec.protocol),
                        "prompt_serialization": prepared_prompt.serialization,
                        "raw_prompt_sha256": raw_prompt_sha256,
                        "prompt_sha256": prompt_sha256,
                        "input_token_sha256": input_token_sha256,
                        "fullkv_text": prediction,
                        "cache_path_text": parity_generation.text,
                        "fullkv_token_ids": list(full_generation.token_ids),
                        "cache_path_token_ids": list(parity_generation.token_ids),
                        "text_match": prediction == parity_generation.text,
                        "token_match": (
                            full_generation.token_ids == parity_generation.token_ids
                        ),
                    }
                    parity_records.append(parity_record)
                    report(
                        "  [parity] "
                        f"text_match={parity_record['text_match']} "
                        f"token_match={parity_record['token_match']}"
                    )

                chunk_size_metrics = _chunk_size_metrics(prefill.chunks)
                full_judgment = _judge_prediction(spec, prediction, gold)
                if "fullkv" in experiment_methods:
                    full_metrics = compute_cache_metrics(
                        sample_id=sample_id,
                        original_length=prefill.sequence_length,
                        budget=prefill.sequence_length,
                        kept_length=prefill.sequence_length,
                        latency_ms=0.0,
                    )
                    method_metric_rows["fullkv"].append(full_metrics)
                    if gold is not None:
                        _append_method_quality(
                            method_qa_rows, "fullkv", prediction, gold, spec.name
                        )
                    runs.append(
                        {
                            "status": "ok",
                            "method": "fullkv",
                            "model": model_name,
                            "dataset": spec.name,
                            "sample_id": sample_id,
                            "model_revision": model_revision,
                            "protocol": canonicalize_protocol(spec.protocol),
                            "prompt_serialization": prepared_prompt.serialization,
                            "raw_prompt_sha256": raw_prompt_sha256,
                            "prompt_sha256": prompt_sha256,
                            "input_token_sha256": input_token_sha256,
                            "config": {
                                "method": "fullkv",
                                "budget": prefill.sequence_length,
                                "budget_type": "fullkv",
                                "budget_value": None,
                                "attention_mode": attention_mode,
                                "layer_index": int(layer_index),
                                "attention_collection": "blockwise",
                                "min_chunk_tokens": int(min_chunk_tokens),
                                "max_chunk_tokens": int(max_chunk_tokens),
                                "min_budget_utilization": float(
                                    min_budget_utilization
                                ),
                                "max_budget_shortfall_tokens": int(
                                    max_budget_shortfall_tokens
                                ),
                                "prefill_block_size": int(prefill_block_size),
                                "protocol": canonicalize_protocol(spec.protocol),
                                "prompt_serialization": prepared_prompt.serialization,
                                "max_length": max_length,
                                "max_new_tokens": int(max_new_tokens),
                                "do_sample": False,
                            },
                            "sequence_length": prefill.sequence_length,
                            "prefill_blocks": prefill.prefill_blocks,
                            "num_chunks": len(prefill.chunks),
                            **chunk_size_metrics,
                            "partially_evicted_chunks": 0,
                            "tier_counts": None,
                            "kept_tokens": prefill.sequence_length,
                            "removed_tokens": 0,
                            "metrics": full_metrics.to_dict(),
                            "prediction": prediction,
                            "evicted_prediction": prediction,
                            "generated_token_ids": list(full_generation.token_ids),
                            "gold": gold,
                            "judgment": full_judgment,
                        }
                    )

                if not compressed_methods:
                    continue

                resolved_budget_configurations = resolve_budget_configurations(
                    prefill.sequence_length,
                    budgets=budgets,
                    budget_ratios=budget_ratios,
                )

                for alpha, recent_window in product(alphas, recent_windows):
                    actual_window = min(max(1, int(recent_window)), prefill.sequence_length)
                    attention_obs = _slice_attention_window(
                        prefill.attention_obs, actual_window
                    )
                    try:
                        scorer = DualSignalScorer(
                            alpha=float(alpha),
                            beta=1.0 - float(alpha),
                            window_size=actual_window,
                            num_layers=(
                                int(attention_obs.shape[0])
                                if attention_obs.ndim == 4
                                else None
                            ),
                            device="cpu",
                        )
                        score_result = scorer.forward_with_details(
                            attention_obs,
                            prefill.chunks,
                            dependency_graph=prefill.dependency_graph,
                        )
                        chunk_scores = score_result.chunk_scores
                        protection_scores = score_result.dependency_scores
                    except Exception as exc:
                        if not continue_on_error:
                            raise
                        config = {
                            "alpha": float(alpha),
                            "beta": 1.0 - float(alpha),
                            "recent_window": int(recent_window),
                        }
                        runs.append(
                            _error_row(
                                model_name=model_name,
                                dataset_name=spec.name,
                                sample_id=sample_id,
                                error=exc,
                                config=config,
                            )
                        )
                        report(f"  [error] {type(exc).__name__}: {exc}")
                        continue

                    for budget_configuration, theta in product(
                        resolved_budget_configurations,
                        thetas,
                    ):
                        budget = int(budget_configuration["budget"])
                        budget_descriptor = budget_configuration["descriptor"]
                        for method in compressed_methods:
                            config = {
                                "method": method,
                                "budget": int(budget),
                                "budget_type": budget_descriptor["type"],
                                "budget_value": budget_descriptor["value"],
                                "budget_specifications": budget_configuration[
                                    "specifications"
                                ],
                                "theta": float(theta),
                                "recent_window": int(recent_window),
                                "alpha": float(alpha),
                                "beta": 1.0 - float(alpha),
                                "attention_mode": attention_mode,
                                "layer_index": int(layer_index),
                                "dependency_top_k": int(dependency_top_k),
                                "attention_collection": "blockwise",
                                "min_chunk_tokens": int(min_chunk_tokens),
                                "max_chunk_tokens": int(max_chunk_tokens),
                                "min_budget_utilization": float(
                                    min_budget_utilization
                                ),
                                "max_budget_shortfall_tokens": int(
                                    max_budget_shortfall_tokens
                                ),
                                "prefill_block_size": int(prefill_block_size),
                                "tier1_score_mode": tier1_score_mode,
                                "protocol": canonicalize_protocol(spec.protocol),
                                "prompt_serialization": prepared_prompt.serialization,
                                "max_length": max_length,
                                "max_new_tokens": int(max_new_tokens),
                                "do_sample": False,
                            }
                            try:
                                run_started = time.perf_counter()
                                report(
                                    f"  [{method}] budget={budget} "
                                    f"theta={theta:g} alpha={alpha:g} "
                                    f"recent_window={recent_window}"
                                )
                                started = time.perf_counter()
                                eviction, masker_result = _run_eviction_method(
                                    method,
                                    prefill,
                                    budget=int(budget),
                                    theta=float(theta),
                                    recent_window=int(recent_window),
                                    chunk_scores=chunk_scores,
                                    protection_scores=protection_scores,
                                    tier1_score_mode=tier1_score_mode,
                                    attention_obs=attention_obs,
                                    allow_level2_fallback=allow_level2_fallback,
                                )
                                elapsed_ms = (time.perf_counter() - started) * 1000.0
                                metrics = compute_cache_metrics(
                                    sample_id=sample_id,
                                    original_length=prefill.sequence_length,
                                    budget=int(budget),
                                    kept_length=int(eviction.kept_indices.numel()),
                                    latency_ms=elapsed_ms,
                                )
                                validate_budget_contract(
                                    metrics,
                                    min_utilization=float(
                                        min_budget_utilization
                                    ),
                                    max_shortfall_tokens=int(
                                        max_budget_shortfall_tokens
                                    ),
                                )
                                metric_rows.append(metrics)
                                method_metric_rows[method].append(metrics)

                                if prefill.next_token_id is not None and max_new_tokens > 0:
                                    generation_result = generate_text_with_evicted_cache(
                                        model=bundle.model,
                                        tokenizer=bundle.tokenizer,
                                        first_new_token_id=prefill.next_token_id,
                                        max_new_tokens=max_new_tokens,
                                        k_cache=eviction.new_k_cache,
                                        v_cache=eviction.new_v_cache,
                                        original_sequence_length=prefill.sequence_length,
                                        budget=int(budget),
                                        kept_indices=eviction.kept_indices,
                                        chunks=(
                                            prefill.chunks
                                            if method == "tdc_kv"
                                            else None
                                        ),
                                        chunk_scores=(
                                            chunk_scores
                                            if method == "tdc_kv"
                                            else None
                                        ),
                                        mask_tiers=(
                                            masker_result.tiers
                                            if masker_result is not None
                                            else None
                                        ),
                                        recent_window=int(recent_window),
                                        return_details=True,
                                    )
                                    evicted_prediction = generation_result.text
                                    generated_token_ids = list(
                                        generation_result.token_ids
                                    )
                                    decode_cache_summary = (
                                        generation_result.cache_summary
                                    )
                                else:
                                    evicted_prediction = prediction
                                    generated_token_ids = list(
                                        full_generation.token_ids
                                    )
                                    decode_cache_summary = None

                                judgment = _judge_prediction(
                                    spec,
                                    evicted_prediction,
                                    gold,
                                )

                                _append_method_quality(
                                    method_qa_rows,
                                    method,
                                    evicted_prediction,
                                    gold,
                                    spec.name,
                                )
                                if method == "tdc_kv" and gold is not None:
                                    evicted_qa_rows.append(
                                        {
                                            "prediction": evicted_prediction,
                                            "gold": gold,
                                            "dataset": spec.name,
                                        }
                                    )

                                runs.append(
                                    {
                                        "status": "ok",
                                        "method": method,
                                        "model": model_name,
                                        "dataset": spec.name,
                                        "sample_id": sample_id,
                                        "model_revision": model_revision,
                                        "protocol": canonicalize_protocol(spec.protocol),
                                        "prompt_serialization": prepared_prompt.serialization,
                                        "raw_prompt_sha256": raw_prompt_sha256,
                                        "prompt_sha256": prompt_sha256,
                                        "input_token_sha256": input_token_sha256,
                                        "config": config,
                                        "sequence_length": prefill.sequence_length,
                                        "prefill_blocks": prefill.prefill_blocks,
                                        "num_chunks": len(prefill.chunks),
                                        **chunk_size_metrics,
                                        "partially_evicted_chunks": int(
                                            eviction.partially_evicted_chunks
                                        ),
                                        "dependency_edges": (
                                            int(
                                                (
                                                    prefill.dependency_graph.neighbor_indices
                                                    >= 0
                                                ).sum().item()
                                            )
                                            if prefill.dependency_graph is not None
                                            else 0
                                        ),
                                        "tier_counts": (
                                            _tier_counts(masker_result.tiers)
                                            if masker_result is not None
                                            else None
                                        ),
                                        "tier_threshold": (
                                            masker_result.threshold
                                            if masker_result is not None
                                            else None
                                        ),
                                        "tier1_source": (
                                            masker_result.tier1_source
                                            if masker_result is not None
                                            else None
                                        ),
                                        "kept_tokens": int(eviction.kept_indices.numel()),
                                        "removed_tokens": int(eviction.removed_indices.numel()),
                                        "score_min": float(chunk_scores.min().item()),
                                        "score_max": float(chunk_scores.max().item()),
                                        "attention_score_min": float(
                                            score_result.attention_scores.min().item()
                                        ),
                                        "attention_score_max": float(
                                            score_result.attention_scores.max().item()
                                        ),
                                        "dependency_score_min": float(
                                            protection_scores.min().item()
                                        ),
                                        "dependency_score_max": float(
                                            protection_scores.max().item()
                                        ),
                                        "decode_cache_summary": decode_cache_summary,
                                        "metrics": metrics.to_dict(),
                                        "prediction": prediction,
                                        "evicted_prediction": evicted_prediction,
                                        "generated_token_ids": generated_token_ids,
                                        "gold": gold,
                                        "judgment": judgment,
                                    }
                                )
                                report(
                                    f"  [{method}] Complete in "
                                    f"{time.perf_counter() - run_started:.1f}s; "
                                    f"kept={int(eviction.kept_indices.numel())}/{budget}"
                                )
                            except Exception as exc:
                                if not continue_on_error:
                                    raise
                                runs.append(
                                    _error_row(
                                        model_name=model_name,
                                        dataset_name=spec.name,
                                        sample_id=sample_id,
                                        error=exc,
                                        config=config,
                                    )
                                )
                                report(
                                    f"  [{method}] Error: {type(exc).__name__}: {exc}"
                                )

    grouped_results = aggregate_grouped_runs(runs)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": model_names,
        "model_revisions": model_revisions,
        "datasets": [spec.to_dict() for spec in dataset_specs],
        "protocols": {
            spec.name: protocol_metadata(spec.protocol) for spec in dataset_specs
        },
        "grid": {
            "methods": experiment_methods,
            "budgets": budgets,
            "budget_ratios": budget_ratios,
            "thetas": thetas,
            "recent_windows": recent_windows,
            "alphas": alphas,
            "max_samples": max_samples,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "attention_mode": attention_mode,
            "layer_index": layer_index,
            "min_chunk_tokens": min_chunk_tokens,
            "max_chunk_tokens": max_chunk_tokens,
            "min_budget_utilization": min_budget_utilization,
            "max_budget_shortfall_tokens": max_budget_shortfall_tokens,
            "dependency_top_k": dependency_top_k,
            "attention_collection": "blockwise",
            "prefill_block_size": prefill_block_size,
            "tier1_score_mode": tier1_score_mode,
            "allow_level2_fallback": allow_level2_fallback,
            "run_fullkv_parity": run_fullkv_parity,
            "prompt_serialization": prompt_serialization,
        },
        "summary": {
            "total_runs": len(runs),
            "successful_runs": sum(1 for row in runs if row["status"] == "ok"),
            "failed_runs": sum(1 for row in runs if row["status"] == "error"),
            "group_count": len(grouped_results),
            "cache_summary": summarize_cache_metrics(metric_rows),
            "baseline_qa_summary": summarize_qa(baseline_qa_rows),
            "evicted_qa_summary": summarize_qa(evicted_qa_rows),
            "fullkv_parity": summarize_generation_parity(parity_records),
            "method_summaries": {
                method: {
                    "cache_summary": summarize_cache_metrics(
                        method_metric_rows.get(method, [])
                    ),
                    "qa_summary": summarize_qa(method_qa_rows.get(method, [])),
                }
                for method in experiment_methods
            },
        },
        "grouped_results": grouped_results,
        "parity_records": parity_records,
        "runs": runs,
    }


__all__ = [
    "DatasetSpec",
    "build_prompt_from_record",
    "load_dataset_records",
    "normalize_methods",
    "parse_dataset_spec",
    "parse_key_value_spec",
    "resolve_budgets",
    "resolve_budget_configurations",
    "run_hf_grid",
]
