#!/usr/bin/env python3
"""Validate generated Parihaka HDF5 files before training or prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


DEFAULT_H5_DIR = "outputs/h5"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", default=DEFAULT_H5_DIR, help="Directory with parihaka_train/val/test.h5 files")
    parser.add_argument(
        "--patch-shape",
        nargs=3,
        type=int,
        required=True,
        metavar=("D", "H", "W"),
        help="Minimum spatial shape required by the configured patch_shape",
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    return parser.parse_args()


def _validate_file(path: Path, patch_shape: tuple[int, int, int]) -> bool:
    if not path.is_file():
        print(f"missing: {path}")
        return False

    ok = True
    with h5py.File(path, "r") as h5:
        for dataset_name in ("raw", "label"):
            if dataset_name not in h5:
                print(f"{path}: missing dataset {dataset_name!r}")
                ok = False
                continue

            shape = tuple(int(v) for v in h5[dataset_name].shape)
            spatial_shape = shape[-3:]
            too_small = [
                f"axis {axis}: data={data_size}, patch={patch_size}"
                for axis, (data_size, patch_size) in enumerate(zip(spatial_shape, patch_shape, strict=True))
                if data_size < patch_size
            ]
            if too_small:
                print(f"{path}:{dataset_name} shape {shape} is smaller than patch_shape {patch_shape}: {too_small}")
                ok = False
            else:
                print(f"{path}:{dataset_name} shape {shape} OK for patch_shape {patch_shape}")

    return ok


def main() -> None:
    args = parse_args()
    h5_dir = Path(args.h5_dir)
    patch_shape = tuple(args.patch_shape)

    ok = True
    for split in args.splits:
        ok = _validate_file(h5_dir / f"parihaka_{split}.h5", patch_shape) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
