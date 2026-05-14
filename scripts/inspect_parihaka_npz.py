#!/usr/bin/env python3
"""Lightweight diagnostics for the local Parihaka-3D NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATA = "../data/ampl3d/parihaka_data.npz"
DEFAULT_LABELS = "../data/ampl3d/parihaka_labels.npz"
FALLBACK_DATA = "../data/ampl_3d/parihaka_data.npz"
FALLBACK_LABELS = "../data/ampl_3d/parihaka_labels.npz"


def _memory_string(nbytes: int) -> str:
    return f"{nbytes / 1024 ** 2:.2f} MiB"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file, got: {path}")


def _resolve_path(path: str, default_path: str, fallback_path: str) -> Path:
    requested = Path(path)
    fallback = Path(fallback_path)
    if requested.exists() or path != default_path or not fallback.exists():
        return requested
    print(f"Default path not found: {requested}")
    print(f"Using local fallback path: {fallback}")
    return fallback


def _print_npz_inventory(path: Path, role: str) -> None:
    _require_file(path)
    print(f"\n[{role}] {path}")
    with np.load(path) as npz:
        print(f"keys: {list(npz.files)}")
        for key in npz.files:
            array = npz[key]
            print(
                f"  - key={key!r}, shape={array.shape}, dtype={array.dtype}, "
                f"memory={_memory_string(array.nbytes)}"
            )
            if role == "amplitude" and np.issubdtype(array.dtype, np.number):
                print(f"    min={np.nanmin(array):.6g}, max={np.nanmax(array):.6g}")
            if role == "labels" and np.issubdtype(array.dtype, np.number):
                unique = np.unique(array)
                print(f"    unique labels: {unique.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to parihaka_data.npz")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to parihaka_labels.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = _resolve_path(args.data, DEFAULT_DATA, FALLBACK_DATA)
    labels_path = _resolve_path(args.labels, DEFAULT_LABELS, FALLBACK_LABELS)
    _print_npz_inventory(data_path, "amplitude")
    _print_npz_inventory(labels_path, "labels")


if __name__ == "__main__":
    main()
