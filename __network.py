"""Public interface for the ExoReL albedo transformer stack."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional

from network_routines.infer import run_inference
from network_routines.train import run_training


def train(par_path: Path | str | Mapping) -> None:
    """Train the transformer using a params JSON file."""
    run_training(par_path)


def infer(par_path: Path | str, p_file: Path | str, lam_file: Path | str, out: Optional[Path | str] = None) -> None:
    """Run inference for a parameter vector and wavelength grid."""
    run_inference(Path(par_path), Path(p_file), Path(lam_file), Path(out) if out else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExoReL neural network routines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--par", type=str, required=True, help="Path to params.json")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--par", type=str, required=True, help="Path to params.json")
    infer_parser.add_argument("--p_file", type=str, required=True, help="Parameter vector file")
    infer_parser.add_argument("--lam_file", type=str, required=True, help="Wavelength file")
    infer_parser.add_argument("--out", type=str, default=None, help="Optional output file for predictions")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args.par)
    else:
        infer(args.par, args.p_file, args.lam_file, args.out)


if __name__ == "__main__":
    main()
