"""HuggingFace-backed experiment runner for TDC-KV."""

from __future__ import annotations

import hashlib
import gc
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
from benchmarks.experiment_identity import build_run_identity, identity_sha256
from benchmarks.gsm8k_protocol import (
    CHUNKKV_GSM8K_8SHOT_PROTOCOL,
    DEFAULT_PROTOCOL,
    build_chunkkv_gsm8k_prompt,
    canonicalize_protocol,
    protocol_metadata,
    validate_protocol_for_adapter,
)
from benchmarks.dataset_manifests import (
    manifest_sha256,
    record_sha256,
    required_record_count,
    select_frozen_records,
)
from benchmarks.model_preflight import (
    enforce_preflight,
    hub_model_preflight,
    loaded_model_preflight,
)
from benchmarks.io_utils import write_json_atomic
from benchmarks.numerical_validation import NumericalIntegrityError, generation_health
from benchmarks.qualification import qualification_report
from benchmarks.checkpoint_store import SQLiteCheckpointStore, is_sqlite_checkpoint
from benchmarks.result_schema import (
    SCHEMA_VERSION,
    assert_result_payload,
    validate_run_row,
)
from benchmarks.reproducibility import collect_environment_metadata, seed_everything
from benchmarks.runtime_metrics import combine_measurements, measure_call
from benchmarks.structural_metrics import (
    compute_evidence_retention,
    compute_head_consensus,
    evidence_texts_from_record,
    locate_evidence_targets,
)
from src.baselines.chunkkv import evict_chunkkv
from src.baselines.h2o import evict_h2o, h2o_token_scores
from src.baselines.snapkv import evict_snapkv, snapkv_token_scores
from src.baselines.streamingllm import evict_streamingllm
from src.core.chunker import FixedSizeChunkConstructor, SentenceBoundaryChunkConstructor
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

METHOD_METADATA = {
    "fullkv": {
        "implementation": "controlled_unpruned_cache",
        "reference_equivalence": "controlled_reference",
        "paper_claim_level": "reference_within_recorded_execution_contract",
        "decode_policy": "controlled_greedy_full_cache",
    },
    "streamingllm": {
        "implementation": "local_sink_plus_recency",
        "reference_equivalence": "approximation",
        "paper_claim_level": "matched_budget_local_approximation",
        "num_sink_tokens": 4,
    },
    "h2o": {
        "implementation": "local_observed_attention_heavy_hitter",
        "reference_equivalence": "approximation",
        "paper_claim_level": "matched_budget_local_approximation",
        "sink_tokens": 1,
        "heavy_hitter_ratio": 1.0,
    },
    "snapkv": {
        "implementation": "local_snapkv_style_observation_scoring",
        "reference_equivalence": "approximation",
        "paper_claim_level": "matched_budget_local_approximation",
        "sink_tokens": 0,
        "kernel_size": 5,
    },
    "chunkkv": {
        "implementation": "local_direct_attention_chunk_selection",
        "reference_equivalence": "approximation",
        "paper_claim_level": "matched_budget_local_approximation",
        "score_source": "direct_attention_only",
    },
    "tdc_kv": {
        "implementation": "repository_tdc_kv",
        "reference_equivalence": "native",
        "paper_claim_level": "proposed_method",
        "score_source": "attention_plus_dependency_routing",
    },
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
    manifest_path = spec.options.get("manifest") or spec.options.get(
        "sample_manifest"
    )
    manifest_partition = spec.options.get("partition")
    if bool(manifest_path) != bool(manifest_partition):
        raise ValueError(
            "Dataset selection requires both `manifest` and `partition` options."
        )
    source_record_limit = max_samples
    if manifest_path and manifest_partition:
        frozen_record_count = required_record_count(
            path=manifest_path,
            spec=spec,
            partition=manifest_partition,
        )
        source_record_limit = max(int(max_samples or 0), frozen_record_count)
    path = Path(spec.source)
    if path.exists():
        records = _load_json_records(path)
    elif spec.adapter == "niah" or spec.source.strip().lower() == "niah":
        generated_count = max_samples
        if manifest_path and manifest_partition:
            generated_count = source_record_limit
        records = _generate_niah_records(spec, max_samples=generated_count)
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
        load_kwargs: dict[str, Any] = {"split": split}
        if spec.options.get("revision"):
            load_kwargs["revision"] = spec.options["revision"]
        if spec.config:
            dataset = load_dataset(dataset_source, spec.config, **load_kwargs)
        else:
            dataset = load_dataset(dataset_source, **load_kwargs)
        if source_record_limit is not None:
            dataset = dataset.select(
                range(min(len(dataset), max(0, int(source_record_limit))))
            )
        records = [dict(row) for row in dataset]

    if manifest_path and manifest_partition:
        records = select_frozen_records(
            records,
            path=manifest_path,
            spec=spec,
            partition=manifest_partition,
        )
    else:
        records = [
            {**record, "__tdc_source_index": index}
            for index, record in enumerate(records)
        ]

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
        # Common standalone words are approximately one token across the target
        # Llama/Mistral/Qwen tokenizers. Numeric suffixes made every nominal
        # "context token" expand into several subword tokens and invalidated
        # the requested NIAH lengths/depths after truncation.
        words.append(word)
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


def _tokenize_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    values = encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))


def materialize_tokenizer_exact_niah_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    """Create a NIAH context with an exact model-tokenizer length.

    NIAH dataset records intentionally store a model-independent nominal
    context.  This function reconstructs that context from tokenizer IDs so the
    requested length and observed needle depth are exact for each target model.
    """
    target = int(record.get("target_context_tokens", 0) or 0)
    if target <= 0:
        raise ValueError("NIAH records require a positive target_context_tokens value.")
    needle = str(record.get("needle") or record.get("answer") or "").strip()
    if not needle:
        raise ValueError("NIAH records require a non-empty needle.")
    depth = float(record.get("needle_depth", 0.5))
    if not math.isfinite(depth) or not 0.0 <= depth <= 1.0:
        raise ValueError("NIAH needle_depth must be in [0, 1].")

    needle_surfaces = (
        f" The secret retrieval key is {needle}.",
        f"The secret retrieval key is {needle}.",
    )
    filler_surfaces = (
        " archive",
        " record",
        " note",
        " section",
        " detail",
        " buffer",
    )

    for filler_surface in filler_surfaces:
        filler_ids = _tokenize_without_special_tokens(tokenizer, filler_surface)
        if len(filler_ids) != 1:
            continue
        filler_id = filler_ids[0]
        for needle_surface in needle_surfaces:
            needle_ids = _tokenize_without_special_tokens(tokenizer, needle_surface)
            if not needle_ids or len(needle_ids) > target:
                continue
            available = target - len(needle_ids)
            prefix_length = min(
                available,
                max(0, int(round(depth * available))),
            )
            context_ids = (
                [filler_id] * prefix_length
                + needle_ids
                + [filler_id] * (available - prefix_length)
            )
            context = _decode_token_ids(tokenizer, context_ids)
            actual_ids = _tokenize_without_special_tokens(tokenizer, context)
            if len(actual_ids) != target:
                continue

            needle_patterns = [
                _tokenize_without_special_tokens(tokenizer, value)
                for value in needle_surfaces
            ]
            starts = [
                start
                for pattern in needle_patterns
                if pattern
                for start in range(len(actual_ids) - len(pattern) + 1)
                if actual_ids[start : start + len(pattern)] == pattern
            ]
            if not starts:
                continue
            needle_start = min(starts)
            materialized = dict(record)
            materialized.update(
                {
                    "context": context,
                    "actual_context_tokens": len(actual_ids),
                    "needle_token_index": needle_start,
                    "actual_needle_depth": needle_start / float(max(1, target - 1)),
                    "tokenizer_exact": True,
                }
            )
            return materialized

    raise RuntimeError(
        "Could not construct a tokenizer-exact NIAH context. The target "
        "tokenizer has no stable one-token filler among the controlled vocabulary."
    )


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


def _cache_memory_metrics(
    *,
    original_k: torch.Tensor,
    original_v: torch.Tensor,
    retained_k: torch.Tensor,
    retained_v: torch.Tensor,
) -> dict[str, int | float]:
    before = int(
        original_k.numel() * original_k.element_size()
        + original_v.numel() * original_v.element_size()
    )
    after = int(
        retained_k.numel() * retained_k.element_size()
        + retained_v.numel() * retained_v.element_size()
    )
    return {
        "kv_bytes_before": before,
        "kv_bytes_after": after,
        "kv_bytes_saved": before - after,
        "kv_memory_retention_ratio": after / float(before) if before else 1.0,
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
    chunkkv_scores: torch.Tensor | None = None,
    baseline_token_scores: torch.Tensor | None = None,
    protect_sink: bool = True,
    protect_recent: bool = True,
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
            protect_sink=bool(protect_sink),
            protect_recent=bool(protect_recent),
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
        independent_scores = (
            chunk_scores if chunkkv_scores is None else chunkkv_scores
        )
        return (
            evict_chunkkv(
                chunk_scores=independent_scores,
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
                token_scores=baseline_token_scores,
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
                token_scores=baseline_token_scores,
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


def _generation_health(
    tokenizer: Any,
    token_ids: Iterable[int],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    special_ids: set[int] = set()
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, (list, tuple, set)):
            special_ids.update(int(item) for item in value if item is not None)
        elif value is not None:
            special_ids.add(int(value))
    return generation_health(
        token_ids,
        max_new_tokens=max_new_tokens,
        special_token_ids=special_ids,
    )


def _successful_numerical_health(
    *,
    prefill_blocks: int,
    generated_tokens: int,
) -> dict[str, Any]:
    return {
        "passed": True,
        "first_failure_stage": None,
        "prefill_blocks_checked": int(prefill_blocks),
        "decode_steps_checked": max(int(generated_tokens) - 1, 0),
        "nonfinite_tensor_count": 0,
    }


def _error_row(
    *,
    model_name: str,
    dataset_name: str,
    sample_id: str,
    error: Exception,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "status": "error",
        "model": model_name,
        "dataset": dataset_name,
        "sample_id": sample_id,
        "config": config or {},
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if isinstance(error, NumericalIntegrityError):
        row["numerical_failure"] = error.to_dict()
    return row


def _run_identity(
    *,
    model_name: str,
    model_revision: str | None,
    dataset_name: str,
    sample_id: str,
    record_sha256_value: str,
    raw_prompt_sha256: str,
    prompt_sha256: str,
    input_token_sha256: str,
    method: str,
    config: dict[str, Any],
    experiment_fingerprint: str,
    actual_dtype: str | None,
    actual_attention_backend: str | None,
    seed: int,
) -> dict[str, str | int]:
    return build_run_identity(
        execution_contract={
            "experiment_fingerprint": experiment_fingerprint,
            "model": model_name,
            "model_revision": model_revision,
            "dataset": dataset_name,
            "protocol": config.get("protocol"),
            "prompt_serialization": config.get("prompt_serialization"),
            "actual_dtype": actual_dtype,
            "actual_attention_backend": actual_attention_backend,
            "max_length": config.get("max_length"),
            "max_new_tokens": config.get("max_new_tokens"),
            "prefill_block_size": config.get("prefill_block_size"),
            "attention_mode": config.get("attention_mode"),
            "layer_weighting": config.get("layer_weighting"),
        },
        sample_input={
            "dataset": dataset_name,
            "sample_id": sample_id,
            "record_sha256": record_sha256_value,
            "raw_prompt_sha256": raw_prompt_sha256,
            "prompt_sha256": prompt_sha256,
            "input_token_sha256": input_token_sha256,
        },
        method_config={
            "method": method,
            "config": config,
        },
        seed=seed,
    )


def _write_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    runs: list[dict[str, Any]],
    parity_records: list[dict[str, Any]],
    state: str = "in_progress",
) -> None:
    """Atomically persist raw progress so an interrupted grid can resume."""
    write_json_atomic(
        path,
        {
            "schema_version": 2,
            "state": str(state),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "runs": runs,
            "parity_records": parity_records,
        },
    )


def run_hf_grid(
    *,
    model_names: list[str],
    model_revisions: dict[str, str] | None = None,
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
    chunking_strategy: str = "sentence",
    fixed_chunk_size: int = 16,
    min_budget_utilization: float = 0.99,
    max_budget_shortfall_tokens: int = 1,
    dependency_top_k: int = 8,
    prefill_block_size: int = 128,
    tier1_score_mode: str = "dependency",
    layer_weighting: str = "linear",
    protect_sink: bool = True,
    protect_recent: bool = True,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = "eager",
    hf_token: str | None = None,
    allow_level2_fallback: bool = True,
    continue_on_error: bool = True,
    progress: bool = False,
    run_fullkv_parity: bool = False,
    parity_max_samples: int | None = None,
    prompt_serialization: str = "auto",
    truncation_side: str = "right",
    seed: int = 42,
    decode_policy: str = "common_streaming",
    experiment_variant: str = "default",
    sample_shard_index: int = 0,
    sample_shard_count: int = 1,
    require_model_preflight: bool = False,
    preflight_require_cuda: bool = False,
    max_vram_fraction: float = 0.90,
    deterministic: bool = True,
    orchestration_job_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
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
    if int(sample_shard_count) <= 0:
        raise ValueError("sample_shard_count must be positive.")
    if int(sample_shard_index) < 0 or int(sample_shard_index) >= int(
        sample_shard_count
    ):
        raise ValueError("sample_shard_index must be in [0, sample_shard_count).")
    if not 0.0 < float(max_vram_fraction) <= 1.0:
        raise ValueError("max_vram_fraction must be in (0, 1].")
    if max_length is not None and int(max_length) <= 0:
        raise ValueError("max_length must be positive when provided.")
    if prefill_block_size <= 0:
        raise ValueError("prefill_block_size must be positive.")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive.")
    if max_chunk_tokens < min_chunk_tokens:
        raise ValueError(
            "max_chunk_tokens must be greater than or equal to "
            "min_chunk_tokens."
        )
    chunking_strategy = str(chunking_strategy).strip().lower()
    if chunking_strategy not in {"sentence", "fixed", "token"}:
        raise ValueError("chunking_strategy must be `sentence`, `fixed`, or `token`.")
    if int(fixed_chunk_size) <= 0:
        raise ValueError("fixed_chunk_size must be positive.")
    layer_weighting = str(layer_weighting).strip().lower()
    if layer_weighting not in {"linear", "uniform"}:
        raise ValueError("layer_weighting must be `linear` or `uniform`.")
    if not 0.0 <= min_budget_utilization <= 1.0:
        raise ValueError("min_budget_utilization must be in [0, 1].")
    if max_budget_shortfall_tokens < 0:
        raise ValueError("max_budget_shortfall_tokens must be non-negative.")
    if run_fullkv_parity and max_new_tokens <= 0:
        raise ValueError("FullKV parity requires max_new_tokens to be positive.")
    if parity_max_samples is not None and int(parity_max_samples) <= 0:
        raise ValueError("parity_max_samples must be positive when provided.")
    prompt_serialization = str(prompt_serialization).strip().lower()
    if prompt_serialization not in {"auto", "raw", "chat"}:
        raise ValueError("prompt_serialization must be `auto`, `raw`, or `chat`.")
    truncation_side = str(truncation_side).strip().lower()
    if truncation_side not in {"left", "right"}:
        raise ValueError("truncation_side must be `left` or `right`.")
    decode_policy = str(decode_policy).strip().lower()
    if decode_policy not in {"common_streaming", "tdc_native"}:
        raise ValueError(
            "decode_policy must be `common_streaming` or `tdc_native`."
        )
    seed_everything(seed, deterministic=deterministic)
    requested_model_revisions = dict(model_revisions or {})
    environment_metadata = collect_environment_metadata(seed=seed)
    environment_signature = {
        "python": environment_metadata.get("python"),
        "packages": environment_metadata.get("packages"),
        "cuda": environment_metadata.get("cuda"),
        "git_commit": (environment_metadata.get("git") or {}).get("commit"),
    }
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
    dataset_manifest_hashes = {
        spec.name: manifest_sha256(
            spec.options.get("manifest") or spec.options["sample_manifest"]
        )
        for spec in dataset_specs
        if spec.options.get("manifest") or spec.options.get("sample_manifest")
    }

    fingerprint_payload = {
        "models": model_names,
        "requested_model_revisions": requested_model_revisions,
        "datasets": [spec.to_dict() for spec in dataset_specs],
        "dataset_manifest_hashes": dataset_manifest_hashes,
        "budgets": budgets,
        "budget_ratios": budget_ratios,
        "thetas": thetas,
        "recent_windows": recent_windows,
        "alphas": alphas,
        "methods": experiment_methods,
        "max_samples": max_samples,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "prompt_template": prompt_template,
        "attention_mode": attention_mode,
        "layer_index": layer_index,
        "min_chunk_tokens": min_chunk_tokens,
        "max_chunk_tokens": max_chunk_tokens,
        "chunking_strategy": chunking_strategy,
        "fixed_chunk_size": int(fixed_chunk_size),
        "dependency_top_k": dependency_top_k,
        "prefill_block_size": prefill_block_size,
        "tier1_score_mode": tier1_score_mode,
        "layer_weighting": layer_weighting,
        "protect_sink": bool(protect_sink),
        "protect_recent": bool(protect_recent),
        "min_budget_utilization": float(min_budget_utilization),
        "max_budget_shortfall_tokens": int(max_budget_shortfall_tokens),
        "allow_level2_fallback": bool(allow_level2_fallback),
        "device": str(device),
        "dtype": str(dtype),
        "trust_remote_code": bool(trust_remote_code),
        "attn_implementation": attn_implementation,
        "prompt_serialization": prompt_serialization,
        "truncation_side": truncation_side,
        "decode_policy": decode_policy,
        "experiment_variant": str(experiment_variant),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "run_fullkv_parity": bool(run_fullkv_parity),
        "parity_max_samples": parity_max_samples,
        "sample_shard_index": int(sample_shard_index),
        "sample_shard_count": int(sample_shard_count),
        "require_model_preflight": bool(require_model_preflight),
        "preflight_require_cuda": bool(preflight_require_cuda),
        "max_vram_fraction": float(max_vram_fraction),
        "environment_signature": environment_signature,
        "orchestration_job_id": orchestration_job_id,
    }
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

    unsharded_dataset_records = {
        spec.name: load_dataset_records(spec, max_samples=max_samples)
        for spec in dataset_specs
    }
    dataset_records = {
        name: records[int(sample_shard_index) :: int(sample_shard_count)]
        for name, records in unsharded_dataset_records.items()
    }
    actual_sample_manifest = {
        spec.name: [
            {
                "sample_id": str(
                    _get_nested(record, spec.id_field)
                    or f"{spec.name}_{record.get('__tdc_source_index', index)}"
                ),
                "source_index": int(record.get("__tdc_source_index", index)),
                "record_sha256": record_sha256(record),
            }
            for index, record in enumerate(dataset_records[spec.name])
        ]
        for spec in dataset_specs
    }
    fingerprint_payload["actual_sample_manifest"] = actual_sample_manifest
    experiment_fingerprint = _sha256_text(
        json.dumps(fingerprint_payload, sort_keys=True, default=str)
    )

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
    checkpoint_store: SQLiteCheckpointStore | None = None
    if checkpoint is not None and is_sqlite_checkpoint(checkpoint):
        checkpoint_store = SQLiteCheckpointStore(
            checkpoint,
            fingerprint=experiment_fingerprint,
            resume=resume,
        )
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError("Resume requires an existing checkpoint file.")
        saved = (
            checkpoint_store.load()
            if checkpoint_store is not None
            else json.loads(checkpoint.read_text(encoding="utf-8"))
        )
        if checkpoint_store is None and saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Legacy checkpoints are diagnostic-only and cannot be resumed.")
        if saved.get("fingerprint") != experiment_fingerprint:
            raise ValueError(
                "Checkpoint configuration does not match the requested experiment."
            )
        # Successful rows are immutable resume units. Failed rows and parity
        # mismatches are retried and replaced instead of poisoning the resumed
        # final result with stale transient failures.
        runs = [
            row
            for row in saved.get("runs", [])
            if row.get("status") == "ok" and not validate_run_row(row)
        ]
        parity_records = [
            row
            for row in saved.get("parity_records", [])
            if row.get("token_match") is True and row.get("text_match") is True
        ]
    completed_run_keys = {
        str(row["run_key"])
        for row in runs
        if row.get("status") == "ok" and row.get("run_key")
    }
    completed_parity_keys = {
        (str(row.get("model")), str(row.get("dataset")), str(row.get("sample_id")))
        for row in parity_records
        if row.get("token_match") is True and row.get("text_match") is True
    }

    def append_run(row: dict[str, Any]) -> None:
        validation_failures = validate_run_row(row)
        if validation_failures:
            raise ValueError(
                "Refusing to record invalid run row: "
                + "; ".join(validation_failures)
            )
        key = row.get("run_key")
        if key and key in completed_run_keys:
            return
        runs.append(row)
        if row.get("status") == "ok" and key:
            completed_run_keys.add(str(key))
        if checkpoint_store is not None:
            checkpoint_store.append_run(row)
        elif checkpoint is not None:
            _write_checkpoint(
                checkpoint,
                fingerprint=experiment_fingerprint,
                runs=runs,
                parity_records=parity_records,
            )

    def save_parity() -> None:
        if checkpoint_store is not None:
            if parity_records:
                checkpoint_store.upsert_parity(parity_records[-1])
        elif checkpoint is not None:
            _write_checkpoint(
                checkpoint,
                fingerprint=experiment_fingerprint,
                runs=runs,
                parity_records=parity_records,
            )
    resolved_model_revisions: dict[str, str | None] = {}
    model_preflights: dict[str, dict[str, Any]] = {}
    model_load_measurements: dict[str, dict] = {}
    max_observation_window = max(1, max(int(window) for window in recent_windows))

    def report(message: str) -> None:
        if progress:
            print(message, flush=True)

    for model_name in model_names:
        report(f"[model] Loading {model_name}...")
        hub_preflight = None
        if require_model_preflight:
            report(f"[model] Checking Hub access for {model_name}...")
            hub_preflight = hub_model_preflight(
                model_name=model_name,
                revision=requested_model_revisions.get(model_name),
                token=hf_token,
                check_local_accelerator=True,
                require_cuda=preflight_require_cuda,
                max_vram_fraction=max_vram_fraction,
            )
            enforce_preflight(hub_preflight)
        measurement_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if str(device).lower() == "auto"
            else torch.device(device)
        )
        model_load = measure_call(
            lambda: load_hf_model_and_tokenizer(
                model_name,
                revision=requested_model_revisions.get(model_name),
                token=hf_token,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            ),
            device=measurement_device,
        )
        bundle = model_load.value
        model_load_measurements[model_name] = model_load.measurement.to_dict()
        model_revision = _model_revision(bundle) or (
            hub_preflight.get("resolved_revision") if hub_preflight else None
        )
        resolved_model_revisions[model_name] = model_revision
        runtime_preflight = loaded_model_preflight(
            model=bundle.model,
            device=bundle.device,
            required_context=(
                int(max_length) + int(max_new_tokens)
                if max_length is not None
                else None
            ),
            prefill_block_size=prefill_block_size,
            require_cuda=preflight_require_cuda,
            max_vram_fraction=max_vram_fraction,
            requested_dtype=dtype,
            requested_attention_backend=attn_implementation,
            require_unquantized=require_model_preflight,
        )
        preflight = {
            "passed": runtime_preflight.get("passed") is True
            and (hub_preflight is None or hub_preflight.get("passed") is True),
            "issues": [
                *(hub_preflight.get("issues") if hub_preflight else []),
                *runtime_preflight.get("issues", []),
            ],
            "hub": hub_preflight,
            "runtime": runtime_preflight,
        }
        model_preflights[model_name] = preflight
        if require_model_preflight:
            enforce_preflight(preflight)
        report(f"[model] Loaded {model_name}.")
        if chunking_strategy == "sentence":
            chunk_constructor = SentenceBoundaryChunkConstructor(
                tokenizer=bundle.tokenizer,
                min_chunk_tokens=min_chunk_tokens,
                max_chunk_tokens=max_chunk_tokens,
                device="cpu",
            )
        else:
            chunk_constructor = FixedSizeChunkConstructor(
                1 if chunking_strategy == "token" else int(fixed_chunk_size),
                device="cpu",
            )

        for spec in dataset_specs:
            records = dataset_records[spec.name]
            for sample_index, record in enumerate(records):
                model_record = (
                    materialize_tokenizer_exact_niah_record(
                        record,
                        tokenizer=bundle.tokenizer,
                    )
                    if spec.adapter == "niah"
                    else record
                )
                sample_id = str(
                    _get_nested(model_record, spec.id_field)
                    or f"{spec.name}_{model_record.get('__tdc_source_index', sample_index)}"
                )
                source_record_sha256 = record_sha256(record)
                report(
                    f"[sample {sample_index + 1}/{len(records)}] "
                    f"dataset={spec.name} id={sample_id}"
                )
                try:
                    prompt, gold = build_prompt_from_record(
                        model_record, spec, prompt_template=prompt_template
                    )
                    prepared_prompt = prepare_prompt(
                        tokenizer=bundle.tokenizer,
                        prompt=prompt,
                        max_length=max_length,
                        serialization=prompt_serialization,
                        truncation_side=truncation_side,
                    )
                    raw_prompt_sha256 = _sha256_text(prompt)
                    prompt_sha256 = _sha256_text(prepared_prompt.rendered_text)
                    report("  [prefill] Collecting blockwise attention and KV cache...")
                    prefill_call = measure_call(
                        lambda: run_hf_prefill(
                            model=bundle.model,
                            tokenizer=bundle.tokenizer,
                            prompt=prompt,
                            sample_id=sample_id,
                            observation_window=max_observation_window,
                            max_length=max_length,
                            attention_mode=attention_mode,
                            layer_index=layer_index,
                            layer_weighting=layer_weighting,
                            min_chunk_tokens=min_chunk_tokens,
                            max_chunk_tokens=max_chunk_tokens,
                            dependency_top_k=dependency_top_k,
                            prefill_block_size=prefill_block_size,
                            chunk_constructor=chunk_constructor,
                            prepared_prompt=prepared_prompt,
                        ),
                        device=bundle.device,
                    )
                    prefill = prefill_call.value
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
                    evidence_targets = locate_evidence_targets(
                        tokenizer=bundle.tokenizer,
                        input_ids=prefill.input_ids,
                        evidence_texts=evidence_texts_from_record(
                            model_record,
                            adapter=spec.adapter,
                            gold=gold,
                        ),
                    )
                    attention_structure = compute_head_consensus(
                        prefill.attention_obs
                    )
                    if prefill.next_token_id is None and max_new_tokens > 0:
                        raise RuntimeError(
                            "FullKV generation requires a prefill next-token prediction."
                        )
                    report("  [fullkv] Running controlled unpruned cache decode...")
                    full_decode_call = measure_call(
                        lambda: generate_text_with_evicted_cache(
                            model=bundle.model,
                            tokenizer=bundle.tokenizer,
                            first_new_token_id=int(prefill.next_token_id or 0),
                            max_new_tokens=max_new_tokens,
                            k_cache=prefill.k_cache,
                            v_cache=prefill.v_cache,
                            original_sequence_length=prefill.sequence_length,
                            budget=None,
                            return_details=True,
                        ),
                        device=bundle.device,
                    )
                    full_generation = full_decode_call.value
                    prediction = full_generation.text
                    if gold is not None:
                        baseline_qa_rows.append(
                            {
                                "prediction": prediction,
                                "gold": gold,
                                "dataset": spec.name,
                            }
                        )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    append_run(
                        _error_row(
                            model_name=model_name,
                            dataset_name=spec.name,
                            sample_id=sample_id,
                            error=exc,
                        )
                    )
                    continue

                parity_key = (model_name, spec.name, sample_id)
                parity_in_scope = (
                    parity_max_samples is None
                    or sample_index < int(parity_max_samples)
                )
                if (
                    run_fullkv_parity
                    and parity_in_scope
                    and parity_key not in completed_parity_keys
                ):
                    report("  [parity] Running HuggingFace native generation...")
                    parity_call = measure_call(
                        lambda: generate_text(
                            model=bundle.model,
                            tokenizer=bundle.tokenizer,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            max_length=max_length,
                            return_details=True,
                            prepared_prompt=prepared_prompt,
                        ),
                        device=bundle.device,
                    )
                    parity_generation = parity_call.value
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
                        "fullkv_text": parity_generation.text,
                        "cache_path_text": prediction,
                        "fullkv_token_ids": list(parity_generation.token_ids),
                        "cache_path_token_ids": list(full_generation.token_ids),
                        "text_match": parity_generation.text == prediction,
                        "token_match": (
                            parity_generation.token_ids == full_generation.token_ids
                        ),
                        "runtime": combine_measurements(
                            huggingface_generation=parity_call.measurement,
                            controlled_prefill=prefill_call.measurement,
                            controlled_decode=full_decode_call.measurement,
                        ),
                    }
                    parity_records.append(parity_record)
                    if parity_record["text_match"] and parity_record["token_match"]:
                        completed_parity_keys.add(parity_key)
                    save_parity()
                    report(
                        "  [parity] "
                        f"text_match={parity_record['text_match']} "
                        f"token_match={parity_record['token_match']}"
                    )

                chunk_size_metrics = _chunk_size_metrics(prefill.chunks)
                full_judgment = _judge_prediction(spec, prediction, gold)
                full_structural_metrics = compute_evidence_retention(
                    evidence_targets,
                    kept_indices=torch.arange(prefill.sequence_length),
                    chunk_map=prefill.chunk_map,
                    sequence_length=prefill.sequence_length,
                )
                full_structural_metrics.update(attention_structure)
                full_structural_metrics["cache_policy_scope"] = "global_shared_mask"
                full_structural_metrics["layerwise_kept_tokens"] = None
                if "fullkv" in experiment_methods:
                    full_metrics = compute_cache_metrics(
                        sample_id=sample_id,
                        original_length=prefill.sequence_length,
                        budget=prefill.sequence_length,
                        kept_length=prefill.sequence_length,
                        latency_ms=full_decode_call.measurement.elapsed_ms,
                    )
                    method_metric_rows["fullkv"].append(full_metrics)
                    if gold is not None:
                        _append_method_quality(
                            method_qa_rows, "fullkv", prediction, gold, spec.name
                        )
                    full_config = {
                        "method": "fullkv",
                        "requested_model_revision": requested_model_revisions.get(
                            model_name
                        ),
                        "budget": prefill.sequence_length,
                        "budget_type": "fullkv",
                        "budget_value": None,
                        "attention_mode": attention_mode,
                        "layer_index": int(layer_index),
                        "attention_collection": "blockwise",
                        "min_chunk_tokens": int(min_chunk_tokens),
                        "max_chunk_tokens": int(max_chunk_tokens),
                        "chunking_strategy": chunking_strategy,
                        "fixed_chunk_size": int(fixed_chunk_size),
                        "layer_weighting": layer_weighting,
                        "protect_sink": bool(protect_sink),
                        "protect_recent": bool(protect_recent),
                        "min_budget_utilization": float(min_budget_utilization),
                        "max_budget_shortfall_tokens": int(
                            max_budget_shortfall_tokens
                        ),
                        "prefill_block_size": int(prefill_block_size),
                        "protocol": canonicalize_protocol(spec.protocol),
                        "prompt_serialization": prepared_prompt.serialization,
                        "max_length": max_length,
                        "max_new_tokens": int(max_new_tokens),
                        "do_sample": False,
                        "seed": int(seed),
                        "experiment_variant": str(experiment_variant),
                        "method_metadata": METHOD_METADATA["fullkv"],
                    }
                    full_runtime = combine_measurements(
                        prefill=prefill_call.measurement,
                        decode=full_decode_call.measurement,
                    )
                    full_runtime.update(
                        {
                            "comparison_scope": "controlled_shared_attention_prefill",
                            "shared_prefill": True,
                            "method_specific_ms": full_decode_call.measurement.elapsed_ms,
                            "generated_tokens": len(full_generation.token_ids),
                            "decode_tokens_per_second": (
                                len(full_generation.token_ids)
                                / (full_decode_call.measurement.elapsed_ms / 1000.0)
                                if full_decode_call.measurement.elapsed_ms > 0.0
                                else None
                            ),
                            "decode_ms_per_token": (
                                full_decode_call.measurement.elapsed_ms
                                / len(full_generation.token_ids)
                                if full_generation.token_ids
                                else None
                            ),
                        }
                    )
                    full_identity = _run_identity(
                        model_name=model_name,
                        model_revision=model_revision,
                        dataset_name=spec.name,
                        sample_id=sample_id,
                        record_sha256_value=source_record_sha256,
                        raw_prompt_sha256=raw_prompt_sha256,
                        prompt_sha256=prompt_sha256,
                        input_token_sha256=input_token_sha256,
                        method="fullkv",
                        config=full_config,
                        experiment_fingerprint=experiment_fingerprint,
                        actual_dtype=runtime_preflight.get("dominant_parameter_dtype"),
                        actual_attention_backend=runtime_preflight.get(
                            "attention_backend"
                        ),
                        seed=seed,
                    )
                    append_run(
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
                            "prompt_original_tokens": prepared_prompt.original_token_count,
                            "prompt_truncated": prepared_prompt.was_truncated,
                            "truncation_side": prepared_prompt.truncation_side,
                            "dataset_runtime_metadata": {
                                key: model_record.get(key)
                                for key in (
                                    "target_context_tokens",
                                    "actual_context_tokens",
                                    "needle_depth",
                                    "actual_needle_depth",
                                    "needle_token_index",
                                    "tokenizer_exact",
                                )
                                if model_record.get(key) is not None
                            },
                            **full_identity,
                            "record_sha256": source_record_sha256,
                            "config": full_config,
                            "sequence_length": prefill.sequence_length,
                            "prefill_blocks": prefill.prefill_blocks,
                            "num_chunks": len(prefill.chunks),
                            **chunk_size_metrics,
                            "partially_evicted_chunks": 0,
                            "cache_memory": _cache_memory_metrics(
                                original_k=prefill.k_cache,
                                original_v=prefill.v_cache,
                                retained_k=prefill.k_cache,
                                retained_v=prefill.v_cache,
                            ),
                            "tier_counts": None,
                            "evidence_targets": evidence_targets.to_dict(),
                            "structural_metrics": full_structural_metrics,
                            "kept_tokens": prefill.sequence_length,
                            "removed_tokens": 0,
                            "metrics": full_metrics.to_dict(),
                            "runtime": full_runtime,
                            "prediction": prediction,
                            "evicted_prediction": prediction,
                            "generated_token_ids": list(full_generation.token_ids),
                            "generation_health": _generation_health(
                                bundle.tokenizer,
                                full_generation.token_ids,
                                max_new_tokens=max_new_tokens,
                            ),
                            "numerical_health": _successful_numerical_health(
                                prefill_blocks=prefill.prefill_blocks,
                                generated_tokens=len(full_generation.token_ids),
                            ),
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
                            layer_weighting=layer_weighting,
                            device="cpu",
                        )
                        tdc_scoring_call = measure_call(
                            lambda: scorer.forward_with_details(
                                attention_obs,
                                prefill.chunks,
                                dependency_graph=prefill.dependency_graph,
                            ),
                            device="cpu",
                        )
                        score_result = tdc_scoring_call.value
                        chunk_scores = score_result.chunk_scores
                        protection_scores = score_result.dependency_scores

                        direct_scorer = DualSignalScorer(
                            alpha=1.0,
                            beta=0.0,
                            window_size=actual_window,
                            num_layers=(
                                int(attention_obs.shape[0])
                                if attention_obs.ndim == 4
                                else None
                            ),
                            layer_weighting=layer_weighting,
                            device="cpu",
                        )
                        chunkkv_scoring_call = measure_call(
                            lambda: direct_scorer.forward_with_details(
                                attention_obs,
                                prefill.chunks,
                                dependency_graph=None,
                            ),
                            device="cpu",
                        )
                        chunkkv_scores = chunkkv_scoring_call.value.chunk_scores

                        baseline_attention = _baseline_attention(attention_obs)
                        h2o_scoring_call = measure_call(
                            lambda: h2o_token_scores(baseline_attention),
                            device="cpu",
                        )
                        snapkv_scoring_call = measure_call(
                            lambda: snapkv_token_scores(
                                baseline_attention,
                                window_size=actual_window,
                            ),
                            device="cpu",
                        )
                        scoring_calls = {
                            "tdc_kv": tdc_scoring_call,
                            "chunkkv": chunkkv_scoring_call,
                            "h2o": h2o_scoring_call,
                            "snapkv": snapkv_scoring_call,
                        }
                    except Exception as exc:
                        if not continue_on_error:
                            raise
                        config = {
                            "alpha": float(alpha),
                            "beta": 1.0 - float(alpha),
                            "recent_window": int(recent_window),
                        }
                        append_run(
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
                            if method != "tdc_kv" and (
                                float(alpha) != float(alphas[0])
                                or float(theta) != float(thetas[0])
                            ):
                                continue
                            if method == "streamingllm" and int(
                                recent_window
                            ) != int(recent_windows[0]):
                                continue
                            config = {
                                "method": method,
                                "requested_model_revision": requested_model_revisions.get(
                                    model_name
                                ),
                                "budget": int(budget),
                                "budget_type": budget_descriptor["type"],
                                "budget_value": budget_descriptor["value"],
                                "budget_specifications": budget_configuration[
                                    "specifications"
                                ],
                                "theta": (
                                    float(theta) if method == "tdc_kv" else None
                                ),
                                "recent_window": (
                                    None
                                    if method == "streamingllm"
                                    else int(recent_window)
                                ),
                                "alpha": (
                                    float(alpha) if method == "tdc_kv" else None
                                ),
                                "beta": (
                                    1.0 - float(alpha)
                                    if method == "tdc_kv"
                                    else None
                                ),
                                "attention_mode": attention_mode,
                                "layer_index": int(layer_index),
                                "dependency_top_k": int(dependency_top_k),
                                "attention_collection": "blockwise",
                                "min_chunk_tokens": int(min_chunk_tokens),
                                "max_chunk_tokens": int(max_chunk_tokens),
                                "chunking_strategy": chunking_strategy,
                                "fixed_chunk_size": int(fixed_chunk_size),
                                "min_budget_utilization": float(
                                    min_budget_utilization
                                ),
                                "max_budget_shortfall_tokens": int(
                                    max_budget_shortfall_tokens
                                ),
                                "prefill_block_size": int(prefill_block_size),
                                "tier1_score_mode": tier1_score_mode,
                                "layer_weighting": layer_weighting,
                                "protect_sink": bool(protect_sink),
                                "protect_recent": bool(protect_recent),
                                "protocol": canonicalize_protocol(spec.protocol),
                                "prompt_serialization": prepared_prompt.serialization,
                                "max_length": max_length,
                                "max_new_tokens": int(max_new_tokens),
                                "do_sample": False,
                                "seed": int(seed),
                                "decode_policy": decode_policy,
                                "experiment_variant": str(experiment_variant),
                                "method_metadata": METHOD_METADATA[method],
                            }
                            run_identity = _run_identity(
                                model_name=model_name,
                                model_revision=model_revision,
                                dataset_name=spec.name,
                                sample_id=sample_id,
                                record_sha256_value=source_record_sha256,
                                raw_prompt_sha256=raw_prompt_sha256,
                                prompt_sha256=prompt_sha256,
                                input_token_sha256=input_token_sha256,
                                method=method,
                                config=config,
                                experiment_fingerprint=experiment_fingerprint,
                                actual_dtype=runtime_preflight.get(
                                    "dominant_parameter_dtype"
                                ),
                                actual_attention_backend=runtime_preflight.get(
                                    "attention_backend"
                                ),
                                seed=seed,
                            )
                            run_key = str(run_identity["run_key"])
                            if run_key in completed_run_keys:
                                report(f"  [{method}] Resumed: already complete.")
                                continue
                            try:
                                run_started = time.perf_counter()
                                report(
                                    f"  [{method}] budget={budget} "
                                    f"theta={theta:g} alpha={alpha:g} "
                                    f"recent_window={recent_window}"
                                )
                                method_token_scores = (
                                    h2o_scoring_call.value
                                    if method == "h2o"
                                    else snapkv_scoring_call.value
                                    if method == "snapkv"
                                    else None
                                )
                                policy_call = measure_call(
                                    lambda: _run_eviction_method(
                                        method,
                                        prefill,
                                        budget=int(budget),
                                        theta=float(theta),
                                        recent_window=int(recent_window),
                                        chunk_scores=chunk_scores,
                                        chunkkv_scores=chunkkv_scores,
                                        protection_scores=protection_scores,
                                        tier1_score_mode=tier1_score_mode,
                                        attention_obs=attention_obs,
                                        baseline_token_scores=method_token_scores,
                                        allow_level2_fallback=allow_level2_fallback,
                                        protect_sink=protect_sink,
                                        protect_recent=protect_recent,
                                    ),
                                    device=prefill.k_cache.device,
                                )
                                eviction, masker_result = policy_call.value
                                metrics = compute_cache_metrics(
                                    sample_id=sample_id,
                                    original_length=prefill.sequence_length,
                                    budget=int(budget),
                                    kept_length=int(eviction.kept_indices.numel()),
                                    latency_ms=policy_call.measurement.elapsed_ms,
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
                                    use_native_tdc_decode = (
                                        decode_policy == "tdc_native"
                                        and method == "tdc_kv"
                                    )
                                    decode_call = measure_call(
                                        lambda: generate_text_with_evicted_cache(
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
                                                if use_native_tdc_decode
                                                else None
                                            ),
                                            chunk_scores=(
                                                chunk_scores
                                                if use_native_tdc_decode
                                                else None
                                            ),
                                            mask_tiers=(
                                                masker_result.tiers
                                                if use_native_tdc_decode
                                                and masker_result is not None
                                                else None
                                            ),
                                            recent_window=int(recent_window),
                                            return_details=True,
                                        ),
                                        device=bundle.device,
                                    )
                                    generation_result = decode_call.value
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
                                    decode_call = None

                                scoring_call = scoring_calls.get(method)
                                runtime = combine_measurements(
                                    prefill=prefill_call.measurement,
                                    scoring=(
                                        scoring_call.measurement
                                        if scoring_call is not None
                                        else None
                                    ),
                                    policy=policy_call.measurement,
                                    decode=(
                                        decode_call.measurement
                                        if decode_call is not None
                                        else None
                                    ),
                                )
                                decode_elapsed_ms = (
                                    decode_call.measurement.elapsed_ms
                                    if decode_call is not None
                                    else 0.0
                                )
                                runtime["generated_tokens"] = len(generated_token_ids)
                                runtime["comparison_scope"] = (
                                    "controlled_shared_attention_prefill"
                                )
                                runtime["shared_prefill"] = True
                                runtime["method_specific_ms"] = sum(
                                    float((runtime["stages"].get(stage) or {}).get("elapsed_ms", 0.0))
                                    for stage in ("scoring", "policy", "decode")
                                )
                                runtime["decode_tokens_per_second"] = (
                                    len(generated_token_ids)
                                    / (decode_elapsed_ms / 1000.0)
                                    if decode_elapsed_ms > 0.0
                                    else None
                                )
                                runtime["decode_ms_per_token"] = (
                                    decode_elapsed_ms / len(generated_token_ids)
                                    if generated_token_ids
                                    else None
                                )
                                policy_scores = (
                                    chunk_scores
                                    if method == "tdc_kv"
                                    else chunkkv_scores
                                    if method == "chunkkv"
                                    else method_token_scores
                                )

                                judgment = _judge_prediction(
                                    spec,
                                    evicted_prediction,
                                    gold,
                                )
                                structural_metrics = compute_evidence_retention(
                                    evidence_targets,
                                    kept_indices=eviction.kept_indices,
                                    chunk_map=prefill.chunk_map,
                                    sequence_length=prefill.sequence_length,
                                )
                                structural_metrics.update(attention_structure)
                                structural_metrics["cache_policy_scope"] = (
                                    "global_shared_mask"
                                )
                                structural_metrics["layerwise_kept_tokens"] = None

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

                                append_run(
                                    {
                                        "status": "ok",
                                        **run_identity,
                                        "record_sha256": source_record_sha256,
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
                                        "prompt_original_tokens": prepared_prompt.original_token_count,
                                        "prompt_truncated": prepared_prompt.was_truncated,
                                        "truncation_side": prepared_prompt.truncation_side,
                                        "dataset_runtime_metadata": {
                                            key: model_record.get(key)
                                            for key in (
                                                "target_context_tokens",
                                                "actual_context_tokens",
                                                "needle_depth",
                                                "actual_needle_depth",
                                                "needle_token_index",
                                                "tokenizer_exact",
                                            )
                                            if model_record.get(key) is not None
                                        },
                                        "config": config,
                                        "sequence_length": prefill.sequence_length,
                                        "prefill_blocks": prefill.prefill_blocks,
                                        "num_chunks": len(prefill.chunks),
                                        **chunk_size_metrics,
                                        "partially_evicted_chunks": int(
                                            eviction.partially_evicted_chunks
                                        ),
                                        "cache_memory": _cache_memory_metrics(
                                            original_k=prefill.k_cache,
                                            original_v=prefill.v_cache,
                                            retained_k=eviction.new_k_cache,
                                            retained_v=eviction.new_v_cache,
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
                                        "evidence_targets": evidence_targets.to_dict(),
                                        "structural_metrics": structural_metrics,
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
                                        "score_source": METHOD_METADATA[method].get(
                                            "score_source"
                                        ),
                                        "score_min": (
                                            float(policy_scores.min().item())
                                            if policy_scores is not None
                                            and policy_scores.numel() > 0
                                            else None
                                        ),
                                        "score_max": (
                                            float(policy_scores.max().item())
                                            if policy_scores is not None
                                            and policy_scores.numel() > 0
                                            else None
                                        ),
                                        "attention_score_min": (
                                            float(
                                                score_result.attention_scores.min().item()
                                            )
                                            if method == "tdc_kv"
                                            else None
                                        ),
                                        "attention_score_max": (
                                            float(
                                                score_result.attention_scores.max().item()
                                            )
                                            if method == "tdc_kv"
                                            else None
                                        ),
                                        "dependency_score_min": (
                                            float(protection_scores.min().item())
                                            if method == "tdc_kv"
                                            else None
                                        ),
                                        "dependency_score_max": (
                                            float(protection_scores.max().item())
                                            if method == "tdc_kv"
                                            else None
                                        ),
                                        "decode_cache_summary": decode_cache_summary,
                                        "metrics": metrics.to_dict(),
                                        "runtime": runtime,
                                        "prediction": prediction,
                                        "evicted_prediction": evicted_prediction,
                                        "generated_token_ids": generated_token_ids,
                                        "generation_health": _generation_health(
                                            bundle.tokenizer,
                                            generated_token_ids,
                                            max_new_tokens=max_new_tokens,
                                        ),
                                        "numerical_health": _successful_numerical_health(
                                            prefill_blocks=prefill.prefill_blocks,
                                            generated_tokens=len(generated_token_ids),
                                        ),
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
                                append_run(
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

        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Rebuild summaries from raw rows so resumed and uninterrupted runs are
    # statistically identical.
    metric_rows = []
    method_metric_rows = {method: [] for method in experiment_methods}
    method_qa_rows = {method: [] for method in experiment_methods}
    baseline_qa_rows = []
    evicted_qa_rows = []
    for row in runs:
        if row.get("status") != "ok":
            continue
        method = str(row.get("method") or row.get("config", {}).get("method"))
        metrics_payload = row.get("metrics")
        if isinstance(metrics_payload, dict):
            metrics_object = CacheMetrics(**metrics_payload)
            method_metric_rows.setdefault(method, []).append(metrics_object)
            if method != "fullkv":
                metric_rows.append(metrics_object)
        gold = row.get("gold")
        if gold is not None:
            qa_row = {
                "prediction": str(row.get("evicted_prediction", "")),
                "gold": str(gold),
                "dataset": str(row.get("dataset", "")),
            }
            method_qa_rows.setdefault(method, []).append(qa_row)
            if method == "fullkv":
                baseline_qa_rows.append(qa_row)
            if method == "tdc_kv":
                evicted_qa_rows.append(qa_row)

    grouped_results = aggregate_grouped_runs(runs)
    protocol_fingerprint = identity_sha256(
        {
            "protocols": {
                spec.name: protocol_metadata(spec.protocol) for spec in dataset_specs
            },
            "dataset_manifest_hashes": dataset_manifest_hashes,
            "prompt_serialization": prompt_serialization,
            "truncation_side": truncation_side,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "decode_policy": decode_policy,
        }
    )
    successful_count = sum(1 for row in runs if row["status"] == "ok")
    failed_count = sum(1 for row in runs if row["status"] == "error")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": model_names,
        "model_revisions": resolved_model_revisions,
        "requested_model_revisions": requested_model_revisions,
        "model_preflights": model_preflights,
        "environment": environment_metadata,
        "model_load_runtime": model_load_measurements,
        "experiment_fingerprint": experiment_fingerprint,
        "orchestration_job_id": orchestration_job_id,
        "job_fingerprint": experiment_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "coverage": {
            "planned_runs": len(runs),
            "observed_runs": len(runs),
            "successful_runs": successful_count,
            "failed_runs": failed_count,
            "validated_runs": successful_count,
        },
        "datasets": [spec.to_dict() for spec in dataset_specs],
        "dataset_manifest_hashes": dataset_manifest_hashes,
        "sample_manifest": actual_sample_manifest,
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
            "chunking_strategy": chunking_strategy,
            "fixed_chunk_size": int(fixed_chunk_size),
            "min_budget_utilization": min_budget_utilization,
            "max_budget_shortfall_tokens": max_budget_shortfall_tokens,
            "dependency_top_k": dependency_top_k,
            "attention_collection": "blockwise",
            "prefill_block_size": prefill_block_size,
            "tier1_score_mode": tier1_score_mode,
            "layer_weighting": layer_weighting,
            "protect_sink": bool(protect_sink),
            "protect_recent": bool(protect_recent),
            "allow_level2_fallback": allow_level2_fallback,
            "run_fullkv_parity": run_fullkv_parity,
            "parity_max_samples": parity_max_samples,
            "prompt_serialization": prompt_serialization,
            "truncation_side": truncation_side,
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "decode_policy": decode_policy,
            "experiment_variant": str(experiment_variant),
            "sample_shard_index": int(sample_shard_index),
            "sample_shard_count": int(sample_shard_count),
            "dtype": str(dtype),
            "device": str(device),
            "attn_implementation": attn_implementation,
            "trust_remote_code": bool(trust_remote_code),
            "require_model_preflight": bool(require_model_preflight),
            "preflight_require_cuda": bool(preflight_require_cuda),
            "max_vram_fraction": float(max_vram_fraction),
            "method_metadata": {
                method: METHOD_METADATA[method] for method in experiment_methods
            },
        },
        "summary": {
            "total_runs": len(runs),
            "successful_runs": successful_count,
            "failed_runs": failed_count,
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
    payload["summary"]["qualification"] = qualification_report(
        payload,
        require_parity=run_fullkv_parity,
        require_cuda=False,
        require_fullkv_pairing=bool(compressed_methods),
        expected_parity_records=(
            sum(
                min(
                    len(samples),
                    int(parity_max_samples)
                    if parity_max_samples is not None
                    else len(samples),
                )
                for samples in actual_sample_manifest.values()
            )
            * len(model_names)
            if run_fullkv_parity
            else None
        ),
        min_gsm8k_parse_rate=(
            0.9 if str(experiment_variant) == "qualification" else 0.0
        ),
    )
    assert_result_payload(payload, require_complete=True)
    if checkpoint_store is not None:
        checkpoint_store.mark_complete()
        checkpoint_store.close()
    elif checkpoint is not None:
        _write_checkpoint(
            checkpoint,
            fingerprint=experiment_fingerprint,
            runs=runs,
            parity_records=parity_records,
            state="complete",
        )
    return payload


__all__ = [
    "DatasetSpec",
    "build_prompt_from_record",
    "load_dataset_records",
    "materialize_tokenizer_exact_niah_record",
    "normalize_methods",
    "parse_dataset_spec",
    "parse_key_value_spec",
    "resolve_budgets",
    "resolve_budget_configurations",
    "run_hf_grid",
]
