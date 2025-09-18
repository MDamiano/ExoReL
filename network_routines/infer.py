"""Inference utility for the ExoReL albedo transformer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import ExoReLAlbedoDataset
from .model import AlbedoTransformer, ModelConfig
from .utils import ensure_output_tree, load_config, select_device


def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")
    if path.suffix in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if len(arr.files) != 1:
                raise ValueError("NPZ files must contain a single array")
            arr = arr[arr.files[0]]
    else:
        arr = np.loadtxt(path, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        raise ValueError("Loaded vector must be 1D")
    return arr


def prepare_model(dataset: ExoReLAlbedoDataset, device: torch.device, checkpoint: Path) -> AlbedoTransformer:
    model = AlbedoTransformer(param_dim=len(dataset.param_columns), config=ModelConfig())
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference(cfg_path: Path, p_file: Path, lam_file: Path, output_array: Path | None = None) -> np.ndarray:
    cfg = load_config(cfg_path)
    output_dirs = ensure_output_tree(cfg["output_directory"])
    checkpoint = output_dirs["checkpoints"] / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device = select_device(cfg.get("network_training", {}).get("device", "auto"))
    dataset = ExoReLAlbedoDataset(cfg["dataset_dir"], random_lambda_fraction=1.0)
    stats = dataset.parameter_stats()
    model = prepare_model(dataset, device, checkpoint)
    params = load_vector(p_file)
    lam = load_vector(lam_file)
    if params.shape[0] != len(dataset.param_columns):
        raise ValueError(
            f"Parameter vector length {params.shape[0]} does not match model input {len(dataset.param_columns)}"
        )
    minimum = stats.minimum.cpu().numpy()
    scale = stats.scale.cpu().numpy()
    params_tensor = torch.from_numpy((params - minimum) / scale).unsqueeze(0).to(device)
    lam_tensor = torch.from_numpy(lam).unsqueeze(0).to(device)
    mask = torch.ones_like(lam_tensor, dtype=torch.bool)
    with torch.no_grad():
        model.fourier.fit(lam_tensor.squeeze(0).cpu())
        prediction = model(params_tensor, lam_tensor, mask=mask)
    albedo = prediction.squeeze(0).cpu().numpy()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig_path = output_dirs["plots"] / f"inference_{timestamp}.png"
    plt.figure(figsize=(6, 4))
    plt.plot(lam, albedo, label="prediction")
    plt.xlabel("wavelength")
    plt.ylabel("albedo")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    if output_array is not None:
        np.savetxt(output_array, np.vstack([lam, albedo]).T, delimiter=",")
    return albedo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the trained albedo transformer")
    parser.add_argument("--par", type=str, required=True, help="Path to params.json")
    parser.add_argument("--p_file", type=str, required=True, help="Parameter vector file")
    parser.add_argument("--lam_file", type=str, required=True, help="Wavelength grid file")
    parser.add_argument("--out", type=str, default=None, help="Optional output file for predicted albedo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(Path(args.par), Path(args.p_file), Path(args.lam_file), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
