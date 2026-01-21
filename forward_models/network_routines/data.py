"""Dataset utilities for ExoReL Albedo transformer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PKG_DIR = Path(__file__).resolve().parent.parent
WAVELENGTH_DIR = PKG_DIR / "forward_mod" / "Data" / "wl_bins"


@dataclass(frozen=True)
class DatasetStats:
    """Parameter statistics for normalisation."""

    minimum: torch.Tensor
    maximum: torch.Tensor

    @property
    def scale(self) -> torch.Tensor:
        return torch.clamp(self.maximum - self.minimum, min=1e-6)


class ExoReLAlbedoDataset(Dataset):
    """PyTorch dataset binding ExoReL generated spectra."""

    def __init__(
        self,
        dataset_dir: str | Path,
        random_lambda_fraction: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.random_lambda_fraction = float(random_lambda_fraction)
        self.dtype = dtype
        csv_path = self.dataset_dir / "dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"dataset.csv not found in {self.dataset_dir}")

        frame = pd.read_csv(csv_path)
        if "index" in frame.columns:
            frame = frame.drop(columns=["index"])
        self.param_columns = list(frame.columns)
        if not self.param_columns:
            raise ValueError("dataset.csv must include at least one parameter column")

        self.params = torch.tensor(frame.to_numpy(dtype=np.float32), dtype=dtype)
        self.stats = DatasetStats(minimum=self.params.min(dim=0).values, maximum=self.params.max(dim=0).values)
        self.sample_files = self._build_sample_index(len(frame))

    def _build_sample_index(self, n_samples: int) -> List[Path]:
        sample_files: List[Path] = []
        for row_idx in range(n_samples):
            fname = self.dataset_dir / f"sample_{row_idx:07d}.json"
            if not fname.exists():
                raise FileNotFoundError(f"Sample file missing: {fname}")
            sample_files.append(fname)
        return sample_files

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        params = self.params[idx]
        params_norm = (params - self.stats.minimum) / self.stats.scale
        sample = self._load_sample(idx)
        lam = sample["lam"]
        target = sample["target"]
        lam, target = self._maybe_subsample(lam, target)
        return {
            "p": params_norm.to(dtype=self.dtype),
            "lam": lam.to(dtype=self.dtype),
            "target": target.to(dtype=self.dtype),
        }

    def parameter_stats(self) -> DatasetStats:
        return self.stats

    def _maybe_subsample(
        self, lam: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frac = max(0.0, min(1.0, self.random_lambda_fraction))
        if frac >= 0.999:
            return lam, target
        count = lam.shape[0]
        k = max(8, int(round(count * frac)))
        k = min(k, count)
        if k == count:
            return lam, target
        idx = torch.randperm(count)[:k]
        idx, _ = torch.sort(idx)
        return lam.index_select(0, idx), target.index_select(0, idx)

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_path = self.sample_files[idx]
        with sample_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        lam_raw = payload.get("wavelength")
        lam = self._load_lambda(lam_raw)
        spectrum = np.asarray(payload.get("spectrum"), dtype=np.float32)
        if spectrum.ndim != 1:
            raise ValueError(f"Spectrum in {sample_path} must be 1D")
        if lam.shape[0] != spectrum.shape[0]:
            raise ValueError(f"Length mismatch in {sample_path}: λ={lam.shape[0]} vs spectrum={spectrum.shape[0]}")
        lam_tensor = torch.from_numpy(lam)
        spec_tensor = torch.from_numpy(spectrum)
        if not torch.isfinite(lam_tensor).all():
            raise ValueError(f"Non-finite wavelength values in {sample_path}")
        inf_mask = torch.isinf(spec_tensor)
        if inf_mask.any():
            spec_tensor[inf_mask] = 0.0
        if torch.isnan(spec_tensor).any():
            spec_tensor = torch.nan_to_num(spec_tensor, nan=0.0)
        return {"lam": lam_tensor, "target": spec_tensor}

    def _load_lambda(self, lam_raw) -> np.ndarray:
        if isinstance(lam_raw, list):
            lam = np.asarray(lam_raw, dtype=np.float32)
        else:
            lam_path = Path(str(lam_raw)).expanduser()
            candidates: List[Path] = []
            seen: set[str] = set()

            def push_with_suffix(base: Path) -> None:
                base = Path(base)
                key = str(base)
                if key not in seen:
                    seen.add(key)
                    candidates.append(base)
                if base.suffix.lower() != ".dat":
                    extra = base.with_suffix(".dat")
                    extra_key = str(extra)
                    if extra_key not in seen:
                        seen.add(extra_key)
                        candidates.append(extra)

            push_with_suffix(lam_path)
            if not lam_path.is_absolute():
                push_with_suffix(self.dataset_dir / lam_path)
                push_with_suffix(WAVELENGTH_DIR / lam_path)

            lam = None
            for candidate in candidates:
                if candidate.exists():
                    lam = np.loadtxt(candidate, dtype=np.float32)
                    break
            if lam is None:
                raise FileNotFoundError(
                    f"Wavelength file not found. Checked: {', '.join(str(p) for p in candidates)}"
                )
        if lam.ndim == 2:
            cols = lam.shape[1]
            if cols == 1:
                lam = lam[:, 0]
            elif cols == 2:
                lam = (lam[:, 0] + lam[:, 1]) * 0.5
            else:
                raise ValueError("Wavelength grid must have one or two columns")
        if lam.ndim != 1:
            raise ValueError("Wavelength grid must be 1D")
        return lam


def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch provided to pad_collate")
    max_len = max(item["lam"].shape[0] for item in batch)
    batch_size = len(batch)
    lam_pad = batch[0]["lam"].new_zeros((batch_size, max_len))
    target_pad = batch[0]["target"].new_zeros((batch_size, max_len))
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=batch[0]["lam"].device)
    params = torch.stack([item["p"] for item in batch], dim=0)
    for i, item in enumerate(batch):
        L = item["lam"].shape[0]
        lam_pad[i, :L] = item["lam"]
        target_pad[i, :L] = item["target"]
        mask[i, :L] = True
    return {"p": params, "lam": lam_pad, "target": target_pad, "mask": mask}
