"""Utility helpers for ExoReL neural network routines."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class TrainingConfig:
    """Container for training hyper-parameters with sensible defaults."""

    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    smoothness_weight: float = 0.0
    patience: int = 10
    seed: int = 42
    random_lambda_fraction: float = 1.0
    device: str = "auto"
    loss: str = "huber"
    huber_delta: float = 0.01
    line_core_fraction: float = 0.0
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    val_fraction: float = 0.1
    dataset_dir: Path = ""
    output_dir: Path = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        if data is None:
            return cls()
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in fields}
        return cls(**kwargs)


def load_config(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Load a configuration JSON file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if "dataset_dir" not in cfg:
        raise KeyError("Configuration must define 'dataset_dir'.")
    if "output_directory" not in cfg:
        raise KeyError("Configuration must define 'output_directory'.")
    cfg.setdefault("network_training", {})
    return cfg


def ensure_output_tree(root: os.PathLike[str] | str) -> Dict[str, Path]:
    """Create output tree with checkpoints, plots, and logs sub-directories."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    checkpoints = root_path / "checkpoints"
    plots = root_path / "plots"
    logs = root_path / "logs"
    checkpoints.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    return {"root": root_path, "checkpoints": checkpoints, "plots": plots, "logs": logs}


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()


def select_device(desired: str = "auto") -> torch.device:
    """Select computation device according to preference order."""

    desired_lower = (desired or "auto").lower()
    candidates: Iterable[Tuple[str, bool]]

    if desired_lower == "mps":
        candidates = (("mps", _mps_available()),)
    elif desired_lower == "cuda":
        candidates = (("cuda", torch.cuda.is_available()),)
    elif desired_lower == "cpu":
        candidates = (("cpu", True),)
    elif desired_lower == "auto":
        candidates = (("mps", _mps_available()), ("cuda", torch.cuda.is_available()), ("cpu", True))
    else:
        logging.warning("Unknown device '%s', falling back to auto-detection.", desired)
        candidates = (("mps", _mps_available()), ("cuda", torch.cuda.is_available()), ("cpu", True))

    for name, available in candidates:
        if available:
            device = torch.device(name)
            print(f"[ExoReL] Using device: {device}")
            return device

    # Fallback if nothing else works
    print("[ExoReL] Falling back to CPU device.")
    return torch.device("cpu")


def setup_logger(log_dir: Path, filename: str = "training.log") -> logging.Logger:
    """Configure a logger that writes to disk and stdout."""

    logger = logging.getLogger("ExoReLNet")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_dir / filename)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def assert_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Raise a ValueError when tensor contains NaNs or Infs."""

    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Persist a dictionary as compact JSON."""

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def cosine_schedule_with_warmup(total_epochs: int, warmup_epochs: int, base_lr: float) -> np.ndarray:
    """Generate per-epoch learning rate multipliers."""

    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    lrs = np.ones(total_epochs, dtype=np.float64)
    if warmup_epochs > 0:
        lrs[:warmup_epochs] = np.linspace(0.1, 1.0, warmup_epochs, endpoint=True)
    remaining = total_epochs - warmup_epochs
    if remaining > 0:
        t = np.linspace(0.0, math.pi, remaining, endpoint=True)
        decay = 0.5 * (1.0 + np.cos(t))
        lrs[warmup_epochs:] = decay
    return lrs * base_lr
