# PLAN:
# - Parse the dataset artifacts produced by GEN_DATASET: a design matrix CSV and sample JSON files.
# - Build SampleRecord objects keyed by dataset index and expose deterministic train/val splits.
# - Load spectra, resolve wavelength grids, and support optional random subsampling for efficiency.
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    index: int
    params: torch.Tensor
    sample_path: Path


class ExoReLAlbedoDataset(Dataset):
    _split_cache: Dict[tuple[str, float, int], Dict[str, List[int]]] = {}

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        random_lambda_fraction: Optional[float] = None,
        seed: int = 0,
        val_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split.lower()
        self.random_lambda_fraction = random_lambda_fraction
        self.rng = random.Random(seed)
        self.val_fraction = val_fraction
        self._wavelength_cache: Dict[str, torch.Tensor] = {}

        manifest = self.dataset_dir / "dataset.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing dataset manifest: {manifest}")
        self.param_names, records_by_index = self._load_design_matrix(manifest)
        self._records_by_index = records_by_index
        self._all_indices = sorted(records_by_index.keys())

        key = (str(self.dataset_dir.resolve()), float(self.val_fraction), int(seed))
        if key not in self._split_cache:
            self._split_cache[key] = self._make_split_indices(self._all_indices, self.val_fraction, seed)
        splits = self._split_cache[key]
        selected = splits.get(self.split, splits.get("all", self._all_indices))
        if not selected:
            raise ValueError(f"No samples available for split '{self.split}' in {self.dataset_dir}")
        self._active_indices = selected
        self.records: List[SampleRecord] = [self._records_by_index[idx] for idx in self._active_indices]

    def _load_design_matrix(self, manifest: Path) -> tuple[List[str], Dict[int, SampleRecord]]:
        with manifest.open("r") as handle:
            header = handle.readline().strip()
        columns = [col.strip() for col in header.split(",") if col]
        data = np.loadtxt(manifest, delimiter=",", skiprows=1)
        if data.size == 0:
            raise ValueError("dataset.csv contains no samples")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        has_index_column = bool(columns) and columns[0].lower() == "index"
        param_offset = 1 if has_index_column else 0
        param_names = columns[param_offset:]
        params = data[:, param_offset:]
        records: Dict[int, SampleRecord] = {}
        for row_idx, row in enumerate(params):
            sample_path = self.dataset_dir / f"sample_{row_idx:07d}.json"
            if not sample_path.exists():
                raise FileNotFoundError(f"Missing sample file: {sample_path}")
            records[row_idx] = SampleRecord(
                index=row_idx,
                params=torch.tensor(row, dtype=torch.float32),
                sample_path=sample_path,
            )
        return param_names, records

    def _make_split_indices(
        self, indices: List[int], val_fraction: float, seed: int
    ) -> Dict[str, List[int]]:
        splits: Dict[str, List[int]] = {"all": list(indices)}
        if val_fraction <= 0 or len(indices) < 2:
            splits["train"] = list(indices)
            splits["val"] = []
            return splits
        rng = random.Random(seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_fraction))
        val_ids = sorted(shuffled[:val_count])
        train_ids = sorted(shuffled[val_count:])
        if not train_ids:
            train_ids, val_ids = val_ids, []
        splits["train"] = train_ids
        splits["val"] = val_ids
        return splits

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_wavelength(self, value) -> torch.Tensor:
        if value is None:
            raise ValueError("Sample JSON missing 'wavelength' entry")
        if isinstance(value, list):
            return torch.tensor(value, dtype=torch.float32)
        if isinstance(value, str):
            if value in self._wavelength_cache:
                return self._wavelength_cache[value]
            candidates = [
                self.dataset_dir / f"{value}.npy",
                self.dataset_dir / f"{value}.json",
                self.dataset_dir / f"{value}.dat",
                Path(__file__).resolve().parent.parent
                / "forward_mod"
                / "Data"
                / "wl_bins"
                / f"{value}.dat",
            ]
            spectrum = None
            for candidate in candidates:
                if candidate.exists():
                    if candidate.suffix == ".npy":
                        spectrum = np.load(candidate)
                    elif candidate.suffix == ".json":
                        spectrum = np.asarray(json.loads(candidate.read_text()), dtype=float)
                    else:
                        spectrum = np.loadtxt(candidate)
                    break
            if spectrum is None:
                raise FileNotFoundError(f"Cannot resolve wavelength grid '{value}'")
            if spectrum.ndim == 2:
                if spectrum.shape[1] == 2:
                    spectrum = spectrum.mean(axis=1)
                else:
                    spectrum = spectrum[:, 2]
            lam = torch.tensor(spectrum, dtype=torch.float32)
            self._wavelength_cache[value] = lam
            return lam
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=torch.float32)
        raise TypeError("Unsupported wavelength representation: %r" % (value,))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | float | None]:
        record = self.records[idx]
        payload = json.loads(record.sample_path.read_text())
        lam = self._resolve_wavelength(payload.get("wavelength"))
        target = torch.tensor(payload["spectrum"], dtype=torch.float32)
        throughput_data = payload.get("throughput")
        throughput = (
            torch.tensor(throughput_data, dtype=torch.float32)
            if throughput_data is not None
            else torch.ones_like(lam)
        )
        lsf_sigma = payload.get("lsf_sigma")
        params = record.params.clone()

        if self.random_lambda_fraction and 0 < self.random_lambda_fraction < 1.0:
            count = max(4, int(len(lam) * self.random_lambda_fraction))
            indices = sorted(self.rng.sample(range(len(lam)), count))
            lam = lam[indices]
            target = target[indices]
            throughput = throughput[indices]

        return {
            "p": params,
            "lam": lam,
            "target": target,
            "throughput": throughput,
            "lsf_sigma": lsf_sigma,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor | float | None]]) -> Dict[str, torch.Tensor | None]:
        batch_size = len(batch)
        lengths = [item["lam"].shape[0] for item in batch]
        max_len = max(lengths)
        lam_tensor = torch.zeros(batch_size, max_len, dtype=torch.float32)
        target_tensor = torch.zeros(batch_size, max_len, dtype=torch.float32)
        throughput_tensor = torch.ones(batch_size, max_len, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        params = torch.stack([item["p"] for item in batch])
        lsf_sigma = batch[0]["lsf_sigma"] if batch[0]["lsf_sigma"] is not None else None
        for i, item in enumerate(batch):
            length = item["lam"].shape[0]
            lam_tensor[i, :length] = item["lam"]
            target_tensor[i, :length] = item["target"]
            throughput_tensor[i, :length] = item["throughput"]
            mask[i, :length] = True
            if item["lsf_sigma"] is not None:
                lsf_sigma = item["lsf_sigma"]
        return {
            "p": params,
            "lam": lam_tensor,
            "target": target_tensor,
            "mask": mask,
            "throughput": throughput_tensor,
            "lsf_sigma": lsf_sigma,
        }


__all__ = ["ExoReLAlbedoDataset", "SampleRecord"]
