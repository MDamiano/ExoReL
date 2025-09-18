# PLAN:
# - Provide helpers to derive suitable Fourier feature frequencies for irregular wavelength grids.
# - Implement a FourierFeatures transformer that normalises wavelengths and constructs sin/cos embeddings.
# - Expose fit/transform style API and stand-alone helper for convenience.
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import math
import numpy as np
import torch

TensorLike = Sequence[float] | np.ndarray | torch.Tensor


def _as_numpy(arr: TensorLike) -> np.ndarray:
    """Convert common tensor types to a contiguous numpy array of floats."""
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float64, copy=False)
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().double().numpy()
    return np.asarray(list(arr), dtype=np.float64)


def derive_frequency_band(
    lam: TensorLike,
    min_period_frac: float = 0.25,
    max_period_frac: float = 1.0,
    safety: float = 0.95,
) -> Tuple[float, float]:
    """Derive a reasonable Fourier frequency band from a wavelength grid.

    Args:
        lam: Wavelength grid of arbitrary spacing.
        min_period_frac: Smallest period as a fraction of the full normalised domain.
        max_period_frac: Largest period as a fraction of the domain.
        safety: Safety factor to keep frequencies within Nyquist.

    Returns:
        (f_min, f_max) for sin/cos features on the normalised grid.
    """
    lam_np = np.sort(_as_numpy(lam))
    if lam_np.size < 2:
        raise ValueError("Need at least two wavelength samples to infer spacing")
    span = float(np.ptp(lam_np))
    u = (lam_np - lam_np.min()) / max(span, 1e-6)
    du = np.diff(u)
    min_du = float(np.clip(du.min(initial=1.0), 1e-6, None))
    # Nyquist frequency (cycles per unit interval) is 0.5 / min spacing.
    nyquist = 0.5 / min_du
    f_max = safety * nyquist * max(min_period_frac, 1e-3)
    f_min = safety * nyquist * max_period_frac * 0.5
    f_min = max(f_min, 0.0)
    f_max = max(f_max, f_min + 1e-6)
    return f_min, f_max


@dataclass
class FourierFeatures:
    num_frequencies: int
    include_raw: bool = True
    learnable_scale: bool = False
    f_min: Optional[float] = None
    f_max: Optional[float] = None

    _lam_min: Optional[float] = None
    _lam_max: Optional[float] = None
    _frequencies: Optional[torch.Tensor] = None
    _scale: Optional[torch.nn.Parameter] = None

    def fit(self, lam: TensorLike) -> "FourierFeatures":
        """Fit normalisation statistics and initialise frequencies."""
        lam_np = _as_numpy(lam)
        lam_min = float(lam_np.min())
        lam_max = float(lam_np.max())
        if not math.isfinite(lam_min) or not math.isfinite(lam_max):
            raise ValueError("Wavelength bounds must be finite")
        if math.isclose(lam_max, lam_min):
            lam_max = lam_min + 1e-6
        self._lam_min = lam_min
        self._lam_max = lam_max
        f_min, f_max = self.f_min, self.f_max
        if f_min is None or f_max is None:
            f_min_auto, f_max_auto = derive_frequency_band(lam_np)
            f_min = f_min if f_min is not None else f_min_auto
            f_max = f_max if f_max is not None else f_max_auto
        freqs = torch.linspace(f_min, f_max, self.num_frequencies, dtype=torch.float32)
        self._frequencies = freqs
        if self.learnable_scale and self._scale is None:
            self._scale = torch.nn.Parameter(torch.ones(1))
        return self

    def _normalise(self, lam: torch.Tensor) -> torch.Tensor:
        if self._lam_min is None or self._lam_max is None:
            raise RuntimeError("FourierFeatures must be fit before transforming")
        return (lam - self._lam_min) / (self._lam_max - self._lam_min)

    def transform(self, lam: TensorLike) -> torch.Tensor:
        """Transform wavelengths into Fourier feature embeddings."""
        if self._frequencies is None:
            raise RuntimeError("Call fit() before transform()")
        lam_tensor = torch.as_tensor(lam, dtype=torch.float32)
        u = self._normalise(lam_tensor).unsqueeze(-1)
        freqs = self._frequencies.to(u.device)
        if self.learnable_scale and self._scale is not None:
            freqs = freqs * self._scale.to(u.device)
        phases = 2 * math.pi * u * freqs
        sin = torch.sin(phases)
        cos = torch.cos(phases)
        components = [sin, cos]
        if self.include_raw:
            components.insert(0, u)
        return torch.cat(components, dim=-1)

    def __call__(self, lam: TensorLike) -> torch.Tensor:
        return self.transform(lam)

    @property
    def output_dim(self) -> int:
        base = 2 * self.num_frequencies
        if self.include_raw:
            base += 1
        return base


__all__ = ["FourierFeatures", "derive_frequency_band"]
