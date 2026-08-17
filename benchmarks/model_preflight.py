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

    gpu_total_memory_bytes = None
    if check_local_accelerator:
        if require_cuda and not torch.cuda.is_available():
            issues.append("paper preflight requires an available CUDA device")
        if torch.cuda.is_available():
            gpu_total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
            if (
                repository_weight_bytes is not None
                and repository_weight_bytes
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

    parameter_bytes = int(
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    )
    element_size = next(model.parameters()).element_size()
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

    attention_backend = getattr(config, "_attn_implementation", None)
    if attention_backend and str(attention_backend).lower() != "eager":
        issues.append(
            f"resolved attention backend is `{attention_backend}`; attention output "
            "support requires eager attention for this paper pipeline"
        )

    return {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "required_context": context or None,
        "context_limit": context_limit,
        "sliding_window": sliding_window,
        "attention_backend": attention_backend,
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
