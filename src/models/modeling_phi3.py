"""Phi-3 integration helpers for TDC-KV and baseline cache policies."""

from __future__ import annotations

from typing import Any

from src.models.modeling_llama import LlamaTDCKVModel, _require_transformers


class Phi3TDCKVModel(LlamaTDCKVModel):
    """Phi-3 specialization of the generic HF TDC-KV wrapper."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
        *,
        tokenizer_name_or_path: str | None = None,
        torch_dtype: Any = "auto",
        device_map: Any = None,
        **model_kwargs: Any,
    ) -> "Phi3TDCKVModel":
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


__all__ = ["Phi3TDCKVModel"]
