"""Fourier feature encoder for wavelength grids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class FourierConfig:
    n_frequencies: int = 16
    max_frequency: float = 128.0
    log_sampling: bool = False
    eps: float = 1e-6
    anneal_steps: int = 0


class FourierFeatures:
    """Fit and transform wavelength arrays into continuous Fourier embeddings."""

    def __init__(self, config: Optional[FourierConfig] = None):
        self.config = config or FourierConfig()
        self._fitted: bool = False
        self._lam_min: float = 0.0
        self._lam_max: float = 1.0
        self._freqs: torch.Tensor | None = None
        self._step: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, lam: torch.Tensor | np.ndarray) -> None:
        """Establish normalisation bounds and optimal frequency range."""

        lam_np = self._to_numpy(lam)
        if lam_np.ndim == 2:
            lam_np = lam_np.reshape(-1)
        lam_np = lam_np[~np.isnan(lam_np)]
        if lam_np.size == 0:
            raise ValueError("Cannot fit FourierFeatures on empty wavelength array")
        lam_min = float(np.min(lam_np))
        lam_max = float(np.max(lam_np))
        if lam_max <= lam_min:
            lam_max = lam_min + 1e-6
        self._lam_min = lam_min
        self._lam_max = lam_max
        max_freq = self._estimate_max_frequency(lam_np)
        geo_freqs = self._geometric_frequencies(max_freq)
        self._freqs = torch.tensor(geo_freqs, dtype=torch.float32)
        self._fitted = True
        self._step = 0

    def transform(
        self,
        lam: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Transform wavelengths into Fourier embeddings."""

        if not self._fitted:
            raise RuntimeError("FourierFeatures must be fitted before calling transform().")

        lam = lam.to(device=device, dtype=dtype)
        u = self._normalize(lam)
        base = [u.unsqueeze(-1)]
        freqs = self._current_freqs(lam.device, dtype)
        phases = 2.0 * math.pi * u.unsqueeze(-1) * freqs
        base.extend([torch.sin(phases), torch.cos(phases)])
        features = torch.cat(base, dim=-1)
        return features

    def step(self) -> None:
        """Advance step for annealing schedule."""

        self._step += 1

    def _normalize(self, lam: torch.Tensor) -> torch.Tensor:
        if self.config.log_sampling:
            lam = torch.log(torch.clamp(lam, min=self.config.eps))
            lam_min = math.log(max(self._lam_min, self.config.eps))
            lam_max = math.log(max(self._lam_max, self._lam_min + self.config.eps))
        else:
            lam_min = self._lam_min
            lam_max = self._lam_max
        denom = max(lam_max - lam_min, self.config.eps)
        return (lam - lam_min) / denom

    def _current_freqs(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self._freqs is not None
        freqs = self._freqs.to(device=device, dtype=dtype)
        if self.config.anneal_steps <= 0:
            return freqs
        alpha = min(1.0, self._step / float(self.config.anneal_steps))
        return freqs * alpha

    def _estimate_max_frequency(self, lam: np.ndarray) -> float:
        lam_sorted = np.sort(lam)
        diffs = np.diff(lam_sorted)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return self.config.max_frequency
        min_spacing = float(np.min(diffs))
        if min_spacing <= 0:
            return self.config.max_frequency
        approximate_max = 0.5 / min_spacing
        return float(np.clip(approximate_max, 1.0, self.config.max_frequency))

    def _geometric_frequencies(self, max_frequency: float) -> np.ndarray:
        n = max(1, self.config.n_frequencies)
        f_min = 1.0
        ratios = np.geomspace(f_min, max_frequency, num=n, endpoint=True)
        return ratios.astype(np.float32)

    @staticmethod
    def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return np.asarray(array)
