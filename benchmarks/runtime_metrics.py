"""CUDA-safe runtime and memory measurements for experiment stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable, Generic, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class StageMeasurement:
    """Wall time and optional CUDA memory statistics for one stage."""

    elapsed_ms: float
    cuda_device: str | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    allocated_after_bytes: int | None
    reserved_after_bytes: int | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MeasuredResult(Generic[T]):
    value: T
    measurement: StageMeasurement


def _cuda_device(device: str | torch.device) -> torch.device | None:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return resolved


def measure_call(
    function: Callable[[], T],
    *,
    device: str | torch.device,
    reset_peak_memory: bool = True,
) -> MeasuredResult[T]:
    """Measure a callable with synchronization on CUDA and no-op memory on CPU."""
    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        if reset_peak_memory:
            torch.cuda.reset_peak_memory_stats(cuda_device)

    started = time.perf_counter()
    value = function()

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if cuda_device is None:
        measurement = StageMeasurement(
            elapsed_ms=elapsed_ms,
            cuda_device=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            allocated_after_bytes=None,
            reserved_after_bytes=None,
        )
    else:
        measurement = StageMeasurement(
            elapsed_ms=elapsed_ms,
            cuda_device=str(cuda_device),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(cuda_device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(cuda_device)),
            allocated_after_bytes=int(torch.cuda.memory_allocated(cuda_device)),
            reserved_after_bytes=int(torch.cuda.memory_reserved(cuda_device)),
        )
    return MeasuredResult(value=value, measurement=measurement)


def combine_measurements(**stages: StageMeasurement | dict | None) -> dict:
    """Build a JSON-safe stage map and total elapsed/maximum peak values."""
    serialized: dict[str, dict] = {}
    for name, measurement in stages.items():
        if measurement is None:
            continue
        serialized[name] = (
            measurement.to_dict()
            if isinstance(measurement, StageMeasurement)
            else dict(measurement)
        )

    elapsed_values = [
        float(row.get("elapsed_ms", 0.0)) for row in serialized.values()
    ]
    allocated_peaks = [
        int(row["peak_allocated_bytes"])
        for row in serialized.values()
        if row.get("peak_allocated_bytes") is not None
    ]
    reserved_peaks = [
        int(row["peak_reserved_bytes"])
        for row in serialized.values()
        if row.get("peak_reserved_bytes") is not None
    ]
    return {
        "stages": serialized,
        "total_measured_ms": sum(elapsed_values),
        "max_peak_allocated_bytes": max(allocated_peaks, default=None),
        "max_peak_reserved_bytes": max(reserved_peaks, default=None),
    }


__all__ = [
    "MeasuredResult",
    "StageMeasurement",
    "combine_measurements",
    "measure_call",
]
