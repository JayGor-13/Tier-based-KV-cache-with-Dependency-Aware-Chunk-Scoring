"""Llama integration helpers for TDC-KV and baseline cache policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.baselines.chunkkv import evict_chunkkv
from src.baselines.h2o import evict_h2o
from src.baselines.snapkv import evict_snapkv
from src.core.chunker import MIN_CHUNK_TOKENS, SentenceBoundaryChunkConstructor, build_punctuation_vocab
from src.core.pipeline import TDCKVPipeline, TDCKVPipelineResult, build_tdc_kv_pipeline
from src.core.scorer import DualSignalScorer
from src.models.cache_utils import (
    apply_keep_indices_to_past_key_values,
    cache_sequence_length,
    evict_past_key_values,
    first_layer_cache_tensors,
    last_layer_observed_attention,
    stack_observed_attention,
)


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for Llama/Phi-3 integration. "
            "Install with: pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):  # pragma: no cover - unlikely
        return torch.device("cpu")


@dataclass(frozen=True)
class CacheCompressionOutput:
    """Outputs from prefill-time cache compression."""

    method: str
    input_ids: Tensor
    attention_obs: Tensor
    compressed_past_key_values: Any
    kept_indices: Tensor
    removed_indices: Tensor
    chunk_scores: Tensor | None = None
    mask_tiers: Tensor | None = None
    chunks: list[Tensor] | None = None
    chunk_map: Tensor | None = None
    pipeline: TDCKVPipelineResult | None = None


class LlamaTDCKVModel:
    """Wrapper around a HF causal LM with TDC-KV prefill compression."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        punct_ids: set[int] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else _model_device(model)
        self.punct_ids = (
            set(punct_ids) if punct_ids is not None else build_punctuation_vocab(tokenizer)
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        torch_dtype: Any = "auto",
        device_map: Any = None,
        **model_kwargs: Any,
    ) -> "LlamaTDCKVModel":
        """Load model/tokenizer and build wrapper.

        Example:
            model = LlamaTDCKVModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        """
        AutoModelForCausalLM, AutoTokenizer = _require_transformers()
        tokenizer_id = tokenizer_name_or_path or model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **model_kwargs,
        )
        return cls(model=model, tokenizer=tokenizer)

    def _chunk_scores(
        self,
        *,
        token_ids: Tensor,
        attention_obs: Tensor,
        min_chunk_tokens: int,
        alpha: float,
        beta: float,
        window_size: int,
    ) -> tuple[list[Tensor], Tensor]:
        chunker = SentenceBoundaryChunkConstructor(
            tokenizer=None,
            punct_ids=self.punct_ids,
            min_chunk_tokens=min_chunk_tokens,
            device=self.device,
        )
        scorer = DualSignalScorer(
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            num_layers=int(attention_obs.shape[0]),
            device=self.device,
        )
        chunks, _ = chunker.forward(token_ids)
        chunk_scores = scorer.forward(attention_obs, chunks)
        return chunks, chunk_scores

    def _compress_with_baseline(
        self,
        *,
        method: str,
        attention_obs: Tensor,
        input_ids: Tensor,
        past_key_values: Any,
        budget: int,
        recent_window: int,
        heavy_hitter_ratio: float,
        min_chunk_tokens: int,
        alpha: float,
        beta: float,
        window_size: int,
    ) -> CacheCompressionOutput:
        k0, v0 = first_layer_cache_tensors(past_key_values)
        method = method.lower()

        chunk_scores = None
        chunks = None
        chunk_map = None
        mask_tiers = None

        if method == "snapkv":
            first_layer_result = evict_snapkv(
                attention_obs=last_layer_observed_attention(
                    attention_obs, window_size=window_size
                ).to(device=k0.device),
                k_cache=k0,
                v_cache=v0,
                budget=budget,
                recent_window=recent_window,
            )
        elif method == "h2o":
            first_layer_result = evict_h2o(
                attention_obs=last_layer_observed_attention(
                    attention_obs, window_size=window_size
                ).to(device=k0.device),
                k_cache=k0,
                v_cache=v0,
                budget=budget,
                recent_window=recent_window,
                heavy_hitter_ratio=heavy_hitter_ratio,
            )
        elif method == "chunkkv":
            chunks, chunk_scores = self._chunk_scores(
                token_ids=input_ids.to(device=self.device, dtype=torch.long),
                attention_obs=attention_obs.to(self.device),
                min_chunk_tokens=min_chunk_tokens,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
            )
            first_layer_result = evict_chunkkv(
                chunk_scores=chunk_scores.to(device=k0.device),
                chunks=chunks,
                k_cache=k0,
                v_cache=v0,
                budget=budget,
                sequence_length=cache_sequence_length(past_key_values),
            )
        else:
            raise ValueError(
                f"Unsupported method `{method}`. "
                "Use one of: tdc_kv, chunkkv, snapkv, h2o."
            )

        compressed = apply_keep_indices_to_past_key_values(
            past_key_values, first_layer_result.kept_indices
        )
        return CacheCompressionOutput(
            method=method,
            input_ids=input_ids,
            attention_obs=attention_obs,
            compressed_past_key_values=compressed,
            kept_indices=first_layer_result.kept_indices,
            removed_indices=first_layer_result.removed_indices,
            chunk_scores=chunk_scores,
            chunks=chunks,
            chunk_map=chunk_map,
            mask_tiers=mask_tiers,
            pipeline=None,
        )

    @torch.no_grad()
    def prefill_and_compress(
        self,
        *,
        prompt: str,
        budget: int,
        method: str = "tdc_kv",
        theta: float = 0.3,
        recent_window: int = 16,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
        alpha: float = 0.6,
        beta: float = 0.4,
        window_size: int = 16,
        heavy_hitter_ratio: float = 0.7,
        allow_level2_fallback: bool = False,
    ) -> CacheCompressionOutput:
        """Run prompt prefill and compress resulting KV cache."""
        method = method.lower()
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        outputs = self.model(
            **model_inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError(
                "Model did not return `past_key_values`. Ensure `use_cache=True`."
            )
        if outputs.attentions is None:
            raise RuntimeError(
                "Model did not return attentions. Ensure `output_attentions=True`."
            )

        attention_obs = stack_observed_attention(
            outputs.attentions, window_size=window_size
        ).to(self.device)
        input_ids = model_inputs["input_ids"][0].to(torch.long)

        if method != "tdc_kv":
            return self._compress_with_baseline(
                method=method,
                attention_obs=attention_obs,
                input_ids=input_ids,
                past_key_values=past_key_values,
                budget=budget,
                recent_window=recent_window,
                heavy_hitter_ratio=heavy_hitter_ratio,
                min_chunk_tokens=min_chunk_tokens,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
            )

        tdc_pipeline = build_tdc_kv_pipeline(
            punct_ids=self.punct_ids,
            min_chunk_tokens=min_chunk_tokens,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            num_layers=int(attention_obs.shape[0]),
            theta=theta,
            recent_window=recent_window,
            allow_level2_fallback=allow_level2_fallback,
            device=self.device,
        )

        k0, v0 = first_layer_cache_tensors(past_key_values)
        pipeline_outputs = tdc_pipeline.run(
            token_ids=input_ids,
            attention_obs=attention_obs,
            k_cache=k0,
            v_cache=v0,
            budget=budget,
            sequence_length=cache_sequence_length(past_key_values),
        )

        full_cache_result = evict_past_key_values(
            past_key_values=past_key_values,
            mask_tiers=pipeline_outputs.mask_tiers.to(self.device),
            chunk_scores=pipeline_outputs.chunk_scores.to(self.device),
            chunks=pipeline_outputs.chunks,
            budget=budget,
            allow_level2_fallback=allow_level2_fallback,
        )
        return CacheCompressionOutput(
            method=method,
            input_ids=input_ids,
            attention_obs=attention_obs,
            compressed_past_key_values=full_cache_result.new_past_key_values,
            kept_indices=full_cache_result.kept_indices,
            removed_indices=full_cache_result.removed_indices,
            chunk_scores=pipeline_outputs.chunk_scores,
            mask_tiers=pipeline_outputs.mask_tiers,
            chunks=pipeline_outputs.chunks,
            chunk_map=pipeline_outputs.chunk_map,
            pipeline=pipeline_outputs,
        )


__all__ = ["CacheCompressionOutput", "LlamaTDCKVModel"]
