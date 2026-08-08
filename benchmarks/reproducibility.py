"""Reproducibility helpers for paper experiment manifests."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed supported random number generators without forcing slow kernels."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(cwd: Path, arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_environment_metadata(
    *, seed: int, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Collect stable software, Git, and accelerator provenance."""
    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    git_status = _git_value(root, ["status", "--porcelain"])
    gpu_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(props.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    return {
        "seed": int(seed),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "numpy",
                "scipy",
                "matplotlib",
            )
        },
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            ),
            "devices": gpu_devices,
        },
        "git": {
            "commit": _git_value(root, ["rev-parse", "HEAD"]),
            "branch": _git_value(root, ["branch", "--show-current"]),
            "dirty": bool(git_status) if git_status is not None else None,
        },
    }


__all__ = ["collect_environment_metadata", "seed_everything"]
