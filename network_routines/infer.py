# PLAN:
# - Provide helpers to load a trained checkpoint and reconstruct the transformer model.
# - Implement a batched prediction utility accepting arbitrary wavelength grids.
# - Offer a CLI demo that plots predictions for a provided parameter vector and wavelength file.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import AlbedoTransformer, AlbedoTransformerConfig


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> AlbedoTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = AlbedoTransformerConfig(**ckpt["model_config"])
    model = AlbedoTransformer(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _prepare_tensor(values: Iterable[float]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)


def predict_albedo(
    model: AlbedoTransformer,
    params: torch.Tensor,
    lam: torch.Tensor,
    throughput: Optional[torch.Tensor] = None,
    lsf_sigma: Optional[float] = None,
) -> torch.Tensor:
    if params.dim() == 1:
        params = params.unsqueeze(0)
    if lam.dim() == 1:
        lam = lam.unsqueeze(0)
    if throughput is not None and throughput.dim() == 1:
        throughput = throughput.unsqueeze(0)
    device = next(model.parameters()).device
    params = params.to(device)
    lam = lam.to(device)
    throughput = throughput.to(device) if throughput is not None else None
    with torch.no_grad():
        pred = model(lam, params, throughput=throughput, lsf_sigma=lsf_sigma)
    return pred.cpu()


def parse_array(path_or_values: str) -> np.ndarray:
    candidate = Path(path_or_values)
    if candidate.exists():
        text = candidate.read_text()
        try:
            data = json.loads(text)
            return np.asarray(data, dtype=np.float32)
        except json.JSONDecodeError:
            return np.loadtxt(candidate, dtype=np.float32)
    return np.asarray([float(x) for x in path_or_values.split(",")], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ExoReL albedo inference")
    parser.add_argument("checkpoint")
    parser.add_argument("--params", required=True, help="Comma-separated values or path to json/txt")
    parser.add_argument("--wavelength", required=True, help="Comma-separated values or path to file")
    parser.add_argument("--throughput", help="Optional throughput values")
    parser.add_argument("--lsf-sigma", type=float)
    parser.add_argument("--output", default="demo_prediction.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device=device)
    params = _prepare_tensor(parse_array(args.params))
    lam = _prepare_tensor(parse_array(args.wavelength))
    throughput = None
    if args.throughput:
        throughput = _prepare_tensor(parse_array(args.throughput))
    pred = predict_albedo(model, params, lam.unsqueeze(0), throughput.unsqueeze(0) if throughput is not None else None, args.lsf_sigma)
    lam_np = lam.numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(lam_np, pred.squeeze(0).numpy(), label="Prediction")
    if throughput is not None:
        plt.plot(lam_np, throughput.numpy(), label="Throughput")
    plt.xlabel("Wavelength")
    plt.ylabel("Albedo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()


if __name__ == "__main__":
    main()
