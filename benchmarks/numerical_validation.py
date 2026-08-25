"""Benchmark-facing numerical and generation health utilities."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable

from src.core.numerical import (
    NumericalIntegrityError,
    TensorHealth,
    require_finite_scalar,
    require_finite_tensor,
    tensor_health,
    validate_token_ids,
)


def generation_health(
    token_ids: Iterable[int],
    *,
    max_new_tokens: int | None = None,
    special_token_ids: Iterable[int] = (),
) -> dict[str, Any]:
    """Return deterministic health signals without judging task correctness."""
    values = tuple(int(value) for value in token_ids)
    counts = Counter(values)
    special = {int(value) for value in special_token_ids}
    non_special = [value for value in values if value not in special]
    non_special_counts = Counter(non_special)
    maximum_count = max(non_special_counts.values(), default=0)
    maximum_fraction = maximum_count / len(non_special) if non_special else 0.0
    reached_limit = (
        max_new_tokens is not None
        and int(max_new_tokens) > 0
        and len(values) >= int(max_new_tokens)
    )
    degenerate_repetition = bool(
        reached_limit and len(non_special) >= 8 and maximum_fraction >= 0.95
    )
    return {
        "decoded_nonempty": bool(values),
        "token_count": len(values),
        "unique_token_count": len(counts),
        "unique_non_special_token_count": len(non_special_counts),
        "maximum_non_special_token_fraction": maximum_fraction,
        "reached_generation_limit": reached_limit,
        "degenerate_repetition": degenerate_repetition,
    }


def find_nonfinite_paths(value: Any, *, path: str = "$") -> list[str]:
    """Return JSON-style paths containing non-finite numeric scalars."""
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(find_nonfinite_paths(child, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            failures.extend(find_nonfinite_paths(child, path=f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    return failures


__all__ = [
    "NumericalIntegrityError",
    "TensorHealth",
    "find_nonfinite_paths",
    "generation_health",
    "require_finite_scalar",
    "require_finite_tensor",
    "tensor_health",
    "validate_token_ids",
]
