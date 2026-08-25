"""Canonical identities for execution, sample input, method, and run rows."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from benchmarks.io_utils import strict_json_dumps


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(child) for child in value]
    if isinstance(value, set):
        return sorted((_normalize(child) for child in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Experiment identity cannot contain NaN or Infinity.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize(dict(payload))


def identity_sha256(payload: Mapping[str, Any]) -> str:
    normalized = canonical_identity_payload(payload)
    encoded = strict_json_dumps(normalized, indent=None).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_identity(
    *,
    execution_contract: Mapping[str, Any],
    sample_input: Mapping[str, Any],
    method_config: Mapping[str, Any],
    seed: int,
    repetition: int = 0,
) -> dict[str, str | int]:
    execution_contract_id = identity_sha256(execution_contract)
    sample_input_id = identity_sha256(sample_input)
    method_config_id = identity_sha256(method_config)
    run_key = identity_sha256(
        {
            "execution_contract_id": execution_contract_id,
            "sample_input_id": sample_input_id,
            "method_config_id": method_config_id,
            "seed": int(seed),
            "repetition": int(repetition),
        }
    )
    return {
        "execution_contract_id": execution_contract_id,
        "sample_input_id": sample_input_id,
        "method_config_id": method_config_id,
        "run_key": run_key,
        "repetition": int(repetition),
    }


__all__ = [
    "build_run_identity",
    "canonical_identity_payload",
    "identity_sha256",
]
