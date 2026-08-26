"""Target-model compatibility and memory preflight checks."""

from __future__ import annotations

from typing import Any

import torch


def hub_model_preflight(
    *,
    model_name: str,
    revision: str | None,
    token: str | None,
    check_local_accelerator: bool = False,
    require_cuda: bool = False,
    max_vram_fraction: float = 0.90,
    quantization: str = "none",
) -> dict[str, Any]:
    """Verify Hub access and resolve the requested ref to an immutable commit."""
    issues: list[str] = []
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            repo_id=model_name,
            revision=revision,
            token=token,
            files_metadata=True,
        )
        resolved_revision = str(info.sha) if info.sha else None
        siblings = list(info.siblings or [])
        safetensor_sizes = [
            int(sibling.size)
            for sibling in siblings
            if str(sibling.rfilename).endswith(".safetensors")
            and sibling.size is not None
        ]
        bin_sizes = [
            int(sibling.size)
            for sibling in siblings
            if str(sibling.rfilename).endswith(".bin") and sibling.size is not None
        ]
        repository_weight_bytes = sum(safetensor_sizes or bin_sizes) or None
    except Exception as exc:  # pragma: no cover - depends on Hub/network state
        resolved_revision = None
        repository_weight_bytes = None
        issues.append(f"{type(exc).__name__}: {exc}")

    quantization_name = str(quantization or "none").strip().lower().replace("_", "-")
    if quantization_name in {"4bit", "bitsandbytes-4bit"}:
        quantization_name = "bnb-4bit"
    estimated_loaded_weight_bytes = repository_weight_bytes
    if repository_weight_bytes is not None and quantization_name == "bnb-4bit":
        estimated_loaded_weight_bytes = int(repository_weight_bytes / 4)

    gpu_total_memory_bytes = None
    if check_local_accelerator:
        if require_cuda and not torch.cuda.is_available():
            issues.append("paper preflight requires an available CUDA device")
        if torch.cuda.is_available():
            gpu_total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
            if (
                estimated_loaded_weight_bytes is not None
                and estimated_loaded_weight_bytes
                > float(max_vram_fraction) * gpu_total_memory_bytes
            ):
                issues.append(
                    "checkpoint weights alone exceed the configured "
                    f"{max_vram_fraction:.0%} VRAM safety limit"
                )

    return {
        "passed": not issues and bool(resolved_revision),
        "issues": issues,
        "model": model_name,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "authenticated": bool(token),
        "repository_weight_bytes": repository_weight_bytes,
        "quantization": quantization_name,
        "estimated_loaded_weight_bytes": estimated_loaded_weight_bytes,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "max_vram_fraction": float(max_vram_fraction),
    }


def _integer(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def loaded_model_preflight(
    *,
    model: Any,
    device: torch.device,
    required_context: int | None,
    prefill_block_size: int,
    require_cuda: bool,
    max_vram_fraction: float = 0.90,
    requested_dtype: str | None = None,
    requested_attention_backend: str | None = None,
    requested_quantization: str = "none",
    requested_quantization_compute_dtype: str | None = None,
    require_unquantized: bool = False,
) -> dict[str, Any]:
    """Validate cache shape assumptions and estimate full-prefill GPU memory."""
    config = model.config
    context_limit = _integer(config, "max_position_embeddings", "n_positions")
    sliding_window = _integer(config, "sliding_window")
    layers = _integer(config, "num_hidden_layers", "n_layer") or 0
    attention_heads = _integer(config, "num_attention_heads", "n_head") or 0
    kv_heads = _integer(config, "num_key_value_heads") or attention_heads
    hidden_size = _integer(config, "hidden_size", "n_embd") or 0
    head_dim = _integer(config, "head_dim") or (
        hidden_size // attention_heads if attention_heads else 0
    )
    context = int(required_context or context_limit or 0)

    issues: list[str] = []
    warnings: list[str] = []
    if require_cuda and device.type != "cuda":
        issues.append("paper preflight requires a CUDA model device")
    if context_limit and context and context > context_limit:
        issues.append(
            f"required context {context} exceeds model capacity {context_limit}"
        )
    if sliding_window and context and sliding_window < context:
        issues.append(
            f"model sliding_window={sliding_window} is below required context {context}; "
            "the current pipeline requires a full prompt cache"
        )

    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("loaded model has no parameters")
    parameter_bytes = int(
        sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    )
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    dtype_bytes: dict[str, int] = {}
    for parameter in parameters:
        name = str(parameter.dtype)
        dtype_bytes[name] = dtype_bytes.get(name, 0) + int(
            parameter.numel() * parameter.element_size()
        )
    dominant_parameter_dtype = max(dtype_bytes, key=dtype_bytes.get)

    attention_backend = getattr(config, "_attn_implementation", None)
    if requested_attention_backend is not None and str(attention_backend) != str(
        requested_attention_backend
    ):
        issues.append(
            "requested attention backend was not resolved: "
            f"requested={requested_attention_backend}, actual={attention_backend}"
        )
    elif attention_backend and str(attention_backend).lower() != "eager":
        issues.append(
            f"resolved attention backend is `{attention_backend}`; attention output "
            "support requires eager attention for this paper pipeline"
        )

    requested_dtype_name = str(requested_dtype or "").strip().lower()
    dtype_aliases = {
        "bf16": "torch.bfloat16",
        "bfloat16": "torch.bfloat16",
        "fp16": "torch.float16",
        "float16": "torch.float16",
        "fp32": "torch.float32",
        "float32": "torch.float32",
    }
    expected_dtype = dtype_aliases.get(requested_dtype_name)

    quantized_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    quantized_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
    quantization_config = getattr(config, "quantization_config", None)
    is_quantized = bool(quantized_4bit or quantized_8bit or quantization_config)
    requested_quantization_name = (
        str(requested_quantization or "none").strip().lower().replace("_", "-")
    )
    if requested_quantization_name in {"4bit", "bitsandbytes-4bit"}:
        requested_quantization_name = "bnb-4bit"
    if requested_quantization_name == "bnb-4bit" and not quantized_4bit:
        issues.append(
            "requested bitsandbytes 4-bit loading was not resolved on the model"
        )
    if requested_quantization_name == "none" and is_quantized:
        issues.append("an unquantized model was requested but a quantized model loaded")
    if require_unquantized and is_quantized:
        issues.append("headline preflight requires an unquantized model")

    def quantization_value(name: str) -> Any:
        if isinstance(quantization_config, dict):
            return quantization_config.get(name)
        return getattr(quantization_config, name, None)

    def canonical_dtype_name(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower().replace("torch.", "")
        return dtype_aliases.get(normalized)

    quantization_compute_dtype = canonical_dtype_name(
        quantization_value("bnb_4bit_compute_dtype")
    )
    requested_compute_dtype = canonical_dtype_name(
        requested_quantization_compute_dtype
    )
    effective_compute_dtype = (
        quantization_compute_dtype if is_quantized else dominant_parameter_dtype
    )
    if is_quantized:
        if (
            requested_compute_dtype is not None
            and quantization_compute_dtype != requested_compute_dtype
        ):
            issues.append(
                "requested quantization compute dtype was not resolved: "
                f"requested={requested_quantization_compute_dtype}, "
                f"actual={quantization_compute_dtype}"
            )
    elif expected_dtype is not None and dominant_parameter_dtype != expected_dtype:
        issues.append(
            "requested dtype was not resolved on model parameters: "
            f"requested={requested_dtype}, dominant={dominant_parameter_dtype}"
        )

    element_sizes = {
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
    }
    element_size = element_sizes.get(
        effective_compute_dtype,
        parameters[0].element_size(),
    )
    kv_bytes = int(2 * layers * kv_heads * head_dim * context * element_size)
    attention_bytes = int(
        layers
        * attention_heads
        * min(max(1, int(prefill_block_size)), max(1, context))
        * context
        * element_size
    )
    estimated_peak = int(1.20 * (parameter_bytes + kv_bytes + attention_bytes))
    gpu_total = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_total = int(torch.cuda.get_device_properties(device).total_memory)
        if estimated_peak > float(max_vram_fraction) * gpu_total:
            issues.append(
                "estimated eager-attention prefill memory exceeds the configured "
                f"{max_vram_fraction:.0%} VRAM safety limit"
            )
    elif not require_cuda:
        warnings.append("CUDA memory fit was not checked on this device")

    bf16_supported = None
    gpu_name = None
    compute_capability = None
    if device.type == "cuda" and torch.cuda.is_available():
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        gpu_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        compute_capability = [int(capability[0]), int(capability[1])]
        if effective_compute_dtype == "torch.bfloat16" and not bf16_supported:
            issues.append("requested BF16 but the CUDA device does not support BF16")

    return {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "required_context": context or None,
        "context_limit": context_limit,
        "sliding_window": sliding_window,
        "attention_backend": attention_backend,
        "requested_attention_backend": requested_attention_backend,
        "requested_dtype": requested_dtype,
        "requested_quantization": requested_quantization_name,
        "requested_quantization_compute_dtype": (
            requested_quantization_compute_dtype
        ),
        "model_config_torch_dtype": str(getattr(config, "torch_dtype", None)),
        "parameter_dtypes": parameter_dtypes,
        "dominant_parameter_dtype": dominant_parameter_dtype,
        "effective_compute_dtype": effective_compute_dtype,
        "quantization_compute_dtype": quantization_compute_dtype,
        "quantization_config": (
            quantization_config.to_dict()
            if hasattr(quantization_config, "to_dict")
            else quantization_config
        ),
        "bf16_supported": bf16_supported,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "is_quantized": is_quantized,
        "loaded_in_4bit": quantized_4bit,
        "loaded_in_8bit": quantized_8bit,
        "parameter_bytes": parameter_bytes,
        "estimated_full_kv_bytes": kv_bytes,
        "estimated_attention_block_bytes": attention_bytes,
        "estimated_peak_bytes_with_margin": estimated_peak,
        "gpu_total_memory_bytes": gpu_total,
        "max_vram_fraction": float(max_vram_fraction),
    }


def enforce_preflight(report: dict[str, Any]) -> None:
    if report.get("passed") is not True:
        raise RuntimeError(
            "Target-model preflight failed: " + "; ".join(report.get("issues") or [])
        )


__all__ = ["enforce_preflight", "hub_model_preflight", "loaded_model_preflight"]
