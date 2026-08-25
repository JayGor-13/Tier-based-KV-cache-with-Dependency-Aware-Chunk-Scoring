"""Strict, atomic serialization helpers for experiment artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def strict_json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    """Serialize RFC-compliant JSON and reject NaN/Infinity."""
    return json.dumps(
        payload,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write strict JSON to a sibling temporary file, then replace atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            strict_json_dumps(payload, indent=indent),
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = ["strict_json_dumps", "write_json_atomic"]
