"""HuggingFace-backed experiment runner for TDC-KV."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import torch

from benchmarks.eval_metrics import (
    CacheMetrics,
    compute_cache_metrics,
    summarize_cache_metrics,
    summarize_qa,
)
from src.baselines.chunkkv import evict_chunkkv
from src.baselines.h2o import evict_h2o
from src.baselines.snapkv import evict_snapkv
from src.baselines.streamingllm import evict_streamingllm
from src.core.chunker import SentenceBoundaryChunkConstructor
from src.core.evictor import EvictionResult, evict_kv_cache
from src.core.masker import assign_protection_tiers
from src.core.scorer import DualSignalScorer
from src.models.cache_utils import (
    generate_text_with_evicted_cache,
    generate_text,
    load_hf_model_and_tokenizer,
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

    source_path = Path(source)
    default_name = source_path.stem if source_path.suffix else source.replace("/", "_")
    return DatasetSpec(
        name=raw.get("name", default_name),
        source=source,
        split=raw.get("split"),
        config=raw.get("config"),
        prompt_field=raw.get("prompt_field", raw.get("prompt", "prompt")),
        answer_field=raw.get("answer_field", raw.get("answer")),
        id_field=raw.get("id_field", raw.get("id")),
        prompt_template=raw.get("template"),
    )


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
    else:
        dataset_source = {
            "gsm8k": "openai/gsm8k",
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

    template = prompt_template or spec.prompt_template or "{prompt}"
    return template.format_map(values), gold


def resolve_budgets(
    sequence_length: int,
    *,
    budgets: Iterable[int],
    budget_ratios: Iterable[float],
) -> list[int]:
    """Resolve absolute and ratio budgets for one tokenized sample."""
    resolved = {
        max(0, min(int(budget), int(sequence_length)))
        for budget in budgets
        if int(budget) >= 0
    }
    for ratio in budget_ratios:
        resolved.add(max(1, min(int(round(float(ratio) * sequence_length)), sequence_length)))
    return sorted(resolved)


def _slice_attention_window(attention_obs: torch.Tensor, window_size: int) -> torch.Tensor:
    actual = min(max(1, int(window_size)), int(attention_obs.shape[-1]))
    return attention_obs[..., -actual:, :]


def _tier_counts(tiers: torch.Tensor) -> dict[str, int]:
    return {
        "tier0": int((tiers == 0).sum().item()),
        "tier1": int((tiers == 1).sum().item()),
        "tier2": int((tiers == 2).sum().item()),
    }


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
    attention_obs: torch.Tensor,
    allow_level2_fallback: bool,
) -> tuple[EvictionResult, torch.Tensor | None]:
    """Apply one compressed-cache method to an HF prefill record."""
    method = normalize_methods([method])[0]
    if method == "fullkv":
        raise ValueError("`fullkv` is handled before eviction.")

    if method == "tdc_kv":
        tiers = assign_protection_tiers(
            chunk_scores=chunk_scores,
            chunks=prefill.chunks,
            theta=float(theta),
            recent_window=int(recent_window),
            sequence_length=prefill.sequence_length,
        )
        eviction = evict_kv_cache(
            mask_tiers=tiers,
            chunk_scores=chunk_scores,
            chunks=prefill.chunks,
            k_cache=prefill.k_cache,
            v_cache=prefill.v_cache,
            budget=int(budget),
            allow_level2_fallback=allow_level2_fallback,
        )
        return eviction, tiers

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
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = "eager",
    allow_level2_fallback: bool = False,
    continue_on_error: bool = True,
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
    experiment_methods = normalize_methods(methods)
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
    max_observation_window = max(int(window) for window in recent_windows)

    for model_name in model_names:
        bundle = load_hf_model_and_tokenizer(
            model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        chunk_constructor = SentenceBoundaryChunkConstructor(
            tokenizer=bundle.tokenizer,
            min_chunk_tokens=min_chunk_tokens,
            device="cpu",
        )

        for spec in dataset_specs:
            for sample_index, record in enumerate(dataset_records[spec.name]):
                sample_id = str(
                    _get_nested(record, spec.id_field) or f"{spec.name}_{sample_index}"
                )
                try:
                    prompt, gold = build_prompt_from_record(
                        record, spec, prompt_template=prompt_template
                    )
                    prediction = generate_text(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        max_length=max_length,
                    )
                    if prediction and gold is not None:
                        baseline_qa_rows.append(
                            {
                                "prediction": prediction,
                                "gold": gold,
                                "dataset": spec.name,
                            }
                        )

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
                        chunk_constructor=chunk_constructor,
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
                        )
                    )
                    continue

                if "fullkv" in experiment_methods:
                    full_metrics = compute_cache_metrics(
                        sample_id=sample_id,
                        original_length=prefill.sequence_length,
                        budget=prefill.sequence_length,
                        kept_length=prefill.sequence_length,
                        latency_ms=0.0,
                    )
                    method_metric_rows["fullkv"].append(full_metrics)
                    if prediction and gold is not None:
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
                            "config": {
                                "method": "fullkv",
                                "budget": prefill.sequence_length,
                                "attention_mode": attention_mode,
                                "layer_index": int(layer_index),
                            },
                            "sequence_length": prefill.sequence_length,
                            "num_chunks": len(prefill.chunks),
                            "tier_counts": None,
                            "kept_tokens": prefill.sequence_length,
                            "removed_tokens": 0,
                            "metrics": full_metrics.to_dict(),
                            "prediction": prediction,
                            "evicted_prediction": prediction,
                            "gold": gold,
                        }
                    )

                if not compressed_methods:
                    continue

                resolved_budgets = resolve_budgets(
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
                        chunk_scores = scorer.forward(attention_obs, prefill.chunks)
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
                        continue

                    for budget, theta in product(resolved_budgets, thetas):
                        for method in compressed_methods:
                            config = {
                                "method": method,
                                "budget": int(budget),
                                "theta": float(theta),
                                "recent_window": int(recent_window),
                                "alpha": float(alpha),
                                "beta": 1.0 - float(alpha),
                                "attention_mode": attention_mode,
                                "layer_index": int(layer_index),
                            }
                            try:
                                started = time.perf_counter()
                                eviction, tiers = _run_eviction_method(
                                    method,
                                    prefill,
                                    budget=int(budget),
                                    theta=float(theta),
                                    recent_window=int(recent_window),
                                    chunk_scores=chunk_scores,
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
                                metric_rows.append(metrics)
                                method_metric_rows[method].append(metrics)

                                if prefill.next_token_id is not None and max_new_tokens > 0:
                                    evicted_prediction = generate_text_with_evicted_cache(
                                        model=bundle.model,
                                        tokenizer=bundle.tokenizer,
                                        first_new_token_id=prefill.next_token_id,
                                        max_new_tokens=max_new_tokens,
                                        k_cache=eviction.new_k_cache,
                                        v_cache=eviction.new_v_cache,
                                        original_sequence_length=prefill.sequence_length,
                                    )
                                else:
                                    evicted_prediction = prediction

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
                                        "config": config,
                                        "sequence_length": prefill.sequence_length,
                                        "num_chunks": len(prefill.chunks),
                                        "tier_counts": (
                                            _tier_counts(tiers)
                                            if tiers is not None
                                            else None
                                        ),
                                        "kept_tokens": int(eviction.kept_indices.numel()),
                                        "removed_tokens": int(eviction.removed_indices.numel()),
                                        "score_min": float(chunk_scores.min().item()),
                                        "score_max": float(chunk_scores.max().item()),
                                        "metrics": metrics.to_dict(),
                                        "prediction": prediction,
                                        "evicted_prediction": evicted_prediction,
                                        "gold": gold,
                                    }
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

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": model_names,
        "datasets": [spec.to_dict() for spec in dataset_specs],
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
            "allow_level2_fallback": allow_level2_fallback,
        },
        "summary": {
            "total_runs": len(runs),
            "successful_runs": sum(1 for row in runs if row["status"] == "ok"),
            "failed_runs": sum(1 for row in runs if row["status"] == "error"),
            "cache_summary": summarize_cache_metrics(metric_rows),
            "baseline_qa_summary": summarize_qa(baseline_qa_rows),
            "evicted_qa_summary": summarize_qa(evicted_qa_rows),
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
    "run_hf_grid",
]
