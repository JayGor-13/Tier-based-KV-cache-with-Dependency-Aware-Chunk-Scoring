"""Shared numerical-integrity checks for model and cache execution.

Paper runs must fail at the first non-finite tensor.  These helpers deliberately
raise instead of sanitizing values because replacing NaN/Inf would make an
invalid eviction policy appear successful.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class TensorHealth:
    """Serializable health summary for one tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    element_count: int
    nonfinite_count: int
    finite_min: float | None
    finite_max: float | None
    finite_abs_max: float | None

    @property
    def finite(self) -> bool:
        return self.nonfinite_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "element_count": self.element_count,
            "nonfinite_count": self.nonfinite_count,
            "finite_min": self.finite_min,
            "finite_max": self.finite_max,
            "finite_abs_max": self.finite_abs_max,
            "finite": self.finite,
        }


class NumericalIntegrityError(FloatingPointError):
    """Raised when a tensor or scalar violates the finite-value contract."""

    def __init__(
        self,
        message: str,
        *,
        tensor_health: TensorHealth | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.tensor_health = tensor_health
        self.context = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": type(self).__name__,
            "message": str(self),
            "context": self.context,
        }
        if self.tensor_health is not None:
            payload["tensor_health"] = self.tensor_health.to_dict()
        return payload


def tensor_health(name: str, tensor: torch.Tensor) -> TensorHealth:
    """Return finite statistics without copying the complete tensor to CPU."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    element_count = int(tensor.numel())
    if element_count == 0 or not (torch.is_floating_point(tensor) or tensor.is_complex()):
        return TensorHealth(
            name=str(name),
            shape=tuple(int(value) for value in tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            element_count=element_count,
            nonfinite_count=0,
            finite_min=None,
            finite_max=None,
            finite_abs_max=None,
        )

    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    nonfinite_count = element_count - int(finite_mask.sum().item())
    finite_values = detached[finite_mask]
    if finite_values.numel() == 0:
        finite_min = finite_max = finite_abs_max = None
    else:
        if finite_values.is_complex():
            magnitudes = finite_values.abs().to(torch.float64)
            finite_min = float(magnitudes.min().item())
            finite_max = float(magnitudes.max().item())
            finite_abs_max = finite_max
        else:
            finite_values = finite_values.to(torch.float64)
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
            finite_abs_max = float(finite_values.abs().max().item())

    return TensorHealth(
        name=str(name),
        shape=tuple(int(value) for value in tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        element_count=element_count,
        nonfinite_count=nonfinite_count,
        finite_min=finite_min,
        finite_max=finite_max,
        finite_abs_max=finite_abs_max,
    )


def require_finite_tensor(
    name: str,
    tensor: torch.Tensor,
    **context: Any,
) -> TensorHealth:
    """Validate a tensor and raise with actionable context on NaN/Inf."""
    health = tensor_health(name, tensor)
    if not health.finite:
        details = ", ".join(f"{key}={value}" for key, value in context.items())
        suffix = f" ({details})" if details else ""
        raise NumericalIntegrityError(
            f"Non-finite values in `{name}`: {health.nonfinite_count}/"
            f"{health.element_count}{suffix}",
            tensor_health=health,
            context=context,
        )
    return health


def require_finite_scalar(name: str, value: Any, **context: Any) -> float:
    """Convert a scalar to float and require a finite value."""
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericalIntegrityError(
            f"`{name}` is not a numeric scalar: {value!r}", context=context
        ) from exc
    if not math.isfinite(converted):
        raise NumericalIntegrityError(
            f"Non-finite scalar `{name}`: {converted!r}", context=context
        )
    return converted


def validate_token_ids(
    token_ids: torch.Tensor | list[int] | tuple[int, ...],
    *,
    vocab_size: int | None,
    name: str = "generated_token_ids",
    **context: Any,
) -> tuple[int, ...]:
    """Require one-dimensional integer IDs inside the known vocabulary."""
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim > 2 or (token_ids.ndim == 2 and token_ids.shape[0] != 1):
            raise ValueError(f"{name} must be a token vector or batch-size-one tensor.")
        if (
            torch.is_floating_point(token_ids)
            or token_ids.is_complex()
            or token_ids.dtype == torch.bool
        ):
            raise NumericalIntegrityError(
                f"`{name}` must use an integer tensor dtype, got {token_ids.dtype}.",
                context=context,
            )
        values = tuple(int(value) for value in token_ids.detach().cpu().flatten().tolist())
    else:
        raw_values = tuple(token_ids)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_values
        ):
            raise NumericalIntegrityError(
                f"`{name}` must contain integer token IDs.", context=context
            )
        values = tuple(int(value) for value in raw_values)
    if any(value < 0 for value in values):
        raise NumericalIntegrityError(
            f"`{name}` contains a negative token ID.", context=context
        )
    if vocab_size is not None and any(value >= int(vocab_size) for value in values):
        raise NumericalIntegrityError(
            f"`{name}` contains a token ID outside vocabulary size {vocab_size}.",
            context=context,
        )
    return values


__all__ = [
    "NumericalIntegrityError",
    "TensorHealth",
    "require_finite_scalar",
    "require_finite_tensor",
    "tensor_health",
    "validate_token_ids",
]
