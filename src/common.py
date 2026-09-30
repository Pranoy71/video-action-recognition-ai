"""Shared utilities: seeding, config loading, device selection.

Design note (the "why"):
Reproducibility is a first-class requirement for this experiment, not an
afterthought. Every training/eval entry point calls `set_global_seed` with the
seed from the config, and we log the resulting environment (torch/python
versions, device, cudnn flags) into every metrics.json so any number in the
README can be traced back to a specific run.
"""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch for reproducible runs.

    `torch.backends.cudnn.deterministic=True` trades a little GPU throughput
    for determinism. For a benchmark-style comparison across three
    architectures, deterministic kernels matter more than raw speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Prefer CUDA, fall back to CPU (the quantization/FPS benchmarks in this
    repo are CPU-first on purpose: edge deployment is where INT8 matters)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config into a plain dict.

    A single flat YAML per experiment (see configs/) keeps the repo auditable:
    every hyperparameter that produced a result is one file, versioned in git.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping")
    return cfg


def env_snapshot() -> Dict[str, Any]:
    """Capture versions/hardware so results are traceable to an environment."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(get_device()),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "os": platform.platform(),
    }


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def find_repo_root() -> Path:
    """Walk up from this file to the repo root (contains configs/ and src/)."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "src" / "data" / "video_dataset.py").exists():
            return parent
    raise RuntimeError("Could not locate repo root (expected src/data/video_dataset.py)")
