"""High-level API for the ExoReL albedo transformer stack."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, Optional

import torch

from .network_routines.infer import load_model as load_trained_model, predict_albedo
from .network_routines.model import AlbedoTransformer, AlbedoTransformerConfig
from .network_routines.train import TrainingConfig, train


def build_model(param_dim: int, overrides: Optional[Dict[str, int | float]] = None) -> AlbedoTransformer:
    """Construct an :class:`AlbedoTransformer` with optional config overrides."""
    overrides = overrides or {}
    config = AlbedoTransformerConfig(param_dim=param_dim, **overrides)
    return AlbedoTransformer(config)


def train_model(config_dict: Dict[str, object]) -> Dict[str, float]:
    """Train the albedo transformer from a raw dictionary of config values."""
    config = TrainingConfig(**config_dict)
    return train(config)


def load_model(checkpoint_path: str, device: str = "cpu") -> AlbedoTransformer:
    """Load a checkpoint produced by :func:`train_model`."""
    return load_trained_model(checkpoint_path, device=device)


def predict(
    model: AlbedoTransformer,
    params: Iterable[float] | torch.Tensor,
    lam: Iterable[float] | torch.Tensor,
    throughput: Optional[Iterable[float] | torch.Tensor] = None,
    lsf_sigma: Optional[float] = None,
) -> torch.Tensor:
    """Convenience wrapper mirroring :func:`network_routines.infer.predict_albedo`."""
    params_tensor = torch.as_tensor(list(params) if not isinstance(params, torch.Tensor) else params)
    lam_tensor = torch.as_tensor(list(lam) if not isinstance(lam, torch.Tensor) else lam)
    throughput_tensor = None
    if throughput is not None:
        throughput_tensor = torch.as_tensor(list(throughput) if not isinstance(throughput, torch.Tensor) else throughput)
    return predict_albedo(model, params_tensor, lam_tensor, throughput_tensor, lsf_sigma)


__all__ = [
    "AlbedoTransformer",
    "AlbedoTransformerConfig",
    "build_model",
    "train_model",
    "TrainingConfig",
    "train",
    "load_model",
    "predict",
]
