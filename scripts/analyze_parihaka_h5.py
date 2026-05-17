#!/usr/bin/env python3
"""Analyze Parihaka HDF5 splits and class coverage in candidate patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_H5_DIR = "outputs/h5_parihaka"
DEFAULT_OUT_JSON = "outputs/diagnostics/parihaka_h5_analysis.json"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", default=DEFAULT_H5_DIR, help="Directory with parihaka_train/val/test.h5 files")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--raw-key", default="raw", help="HDF5 dataset key for seismic amplitudes")
    parser.add_argument("--label-key", default="label", help="HDF5 dataset key for labels")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--patch-shape", nargs=3, type=int, default=[64, 128, 128], metavar=("D", "H", "W"))
    parser.add_argument("--stride-shape", nargs=3, type=int, default=[48, 96, 96], metavar=("D", "H", "W"))
    parser.add_argument("--min-class-voxels", type=int, default=256)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for the JSON diagnostic report")
    return parser.parse_args()


def _require_positive(name: str, values: tuple[int, ...]) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive, got {values}")


def _shape(dataset: h5py.Dataset) -> list[int]:
    return [int(value) for value in dataset.shape]


def _raw_spatial_shape(raw_shape: list[int] | None) -> list[int] | None:
    if raw_shape is None:
        return None
    if len(raw_shape) == 3:
        return raw_shape
    if len(raw_shape) == 4:
        return raw_shape[1:]
    return None


def _raw_channel_count(raw_shape: list[int] | None) -> int | None:
    if raw_shape is None:
        return None
    if len(raw_shape) == 3:
        return 1
    if len(raw_shape) == 4:
        return int(raw_shape[0])
    return None


def _label_value(value: Any) -> int | float | str:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return int(value) if value.is_integer() else float(value)
    return str(value)


def _is_valid_class_value(value: Any, num_classes: int) -> bool:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        return False
    class_id = int(numeric_value)
    return 0 <= class_id < num_classes


def _class_counts_array(values: np.ndarray, num_classes: int) -> np.ndarray:
    if np.issubdtype(values.dtype, np.integer):
        flat = values.reshape(-1)
        valid = flat[(flat >= 0) & (flat < num_classes)]
        if valid.size == 0:
            return np.zeros(num_classes, dtype=np.int64)
        return np.bincount(valid.astype(np.int64, copy=False), minlength=num_classes)[:num_classes]

    return np.array([np.count_nonzero(values == class_id) for class_id in range(num_classes)], dtype=np.int64)


def _gen_starts(size: int, patch_size: int, stride: int) -> list[int]:
    if size < patch_size:
        return []

    starts = list(range(0, size - patch_size + 1, stride))
    last_start = size - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _spatial_label_view(label: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if label.ndim == 3:
        return label, None

    squeezed = np.squeeze(label)
    if squeezed.ndim == 3:
        return squeezed, f"squeezed singleton axes from {tuple(label.shape)} to {tuple(squeezed.shape)}"

    return None, f"patch coverage requires a 3D label volume, got shape {tuple(label.shape)}"


def _empty_patch_coverage(num_classes: int, reason: str) -> dict[str, Any]:
    return {
        "total_patches": 0,
        "skipped_reason": reason,
        "per_class": {
            str(class_id): {
                "patches_with_class": 0,
                "fraction_with_class": 0.0,
                "patches_with_min_voxels": 0,
                "fraction_with_min_voxels": 0.0,
            }
            for class_id in range(num_classes)
        },
    }


def _patch_coverage(
    label: np.ndarray,
    *,
    num_classes: int,
    patch_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
    min_class_voxels: int,
) -> dict[str, Any]:
    spatial_label, note = _spatial_label_view(label)
    if spatial_label is None:
        return _empty_patch_coverage(num_classes, note or "unsupported label shape")

    starts = [
        _gen_starts(int(size), int(patch), int(stride))
        for size, patch, stride in zip(spatial_label.shape, patch_shape, stride_shape, strict=True)
    ]
    if any(len(axis_starts) == 0 for axis_starts in starts):
        reason = f"label shape {tuple(spatial_label.shape)} is smaller than patch_shape {patch_shape}"
        return _empty_patch_coverage(num_classes, reason)

    total_patches = int(np.prod([len(axis_starts) for axis_starts in starts], dtype=np.int64))
    patches_with_class = np.zeros(num_classes, dtype=np.int64)
    patches_with_min_voxels = np.zeros(num_classes, dtype=np.int64)
    p_z, p_y, p_x = patch_shape

    for z_start in starts[0]:
        z_stop = z_start + p_z
        for y_start in starts[1]:
            y_stop = y_start + p_y
            for x_start in starts[2]:
                x_stop = x_start + p_x
                patch = spatial_label[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                counts = _class_counts_array(patch, num_classes)
                patches_with_class += counts > 0
                patches_with_min_voxels += counts >= min_class_voxels

    per_class = {}
    for class_id in range(num_classes):
        with_class = int(patches_with_class[class_id])
        with_min_voxels = int(patches_with_min_voxels[class_id])
        per_class[str(class_id)] = {
            "patches_with_class": with_class,
            "fraction_with_class": with_class / total_patches,
            "patches_with_min_voxels": with_min_voxels,
            "fraction_with_min_voxels": with_min_voxels / total_patches,
        }

    coverage: dict[str, Any] = {
        "total_patches": total_patches,
        "per_class": per_class,
    }
    if note is not None:
        coverage["note"] = note
    return coverage


def _analyze_labels(
    label_dataset: h5py.Dataset,
    *,
    num_classes: int,
    patch_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
    min_class_voxels: int,
) -> dict[str, Any]:
    label = label_dataset[...]
    unique_values = np.unique(label)
    unique_labels = [_label_value(value) for value in unique_values.tolist()]
    invalid_labels = [_label_value(value) for value in unique_values.tolist() if not _is_valid_class_value(value, num_classes)]

    counts_array = _class_counts_array(label, num_classes)
    total_voxels = int(label.size)
    class_counts = {str(class_id): int(counts_array[class_id]) for class_id in range(num_classes)}
    class_fractions = {
        str(class_id): (float(counts_array[class_id]) / total_voxels if total_voxels else 0.0)
        for class_id in range(num_classes)
    }
    missing_classes = [class_id for class_id in range(num_classes) if counts_array[class_id] == 0]

    return {
        "unique_labels": unique_labels,
        "invalid_labels": invalid_labels,
        "missing_classes": missing_classes,
        "class_counts": class_counts,
        "class_fractions": class_fractions,
        "patch_coverage": _patch_coverage(
            label,
            num_classes=num_classes,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            min_class_voxels=min_class_voxels,
        ),
    }


def analyze_split(
    path: Path,
    *,
    raw_key: str,
    label_key: str,
    num_classes: int,
    patch_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
    min_class_voxels: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "raw_shape": None,
        "raw_spatial_shape": None,
        "raw_channels": None,
        "label_shape": None,
        "raw_dtype": None,
        "label_dtype": None,
        "shape_match": False,
    }

    if not path.is_file():
        result["error"] = "file_not_found"
        return result

    with h5py.File(path, "r") as h5:
        raw_dataset = h5.get(raw_key)
        label_dataset = h5.get(label_key)

        if not isinstance(raw_dataset, h5py.Dataset):
            result.setdefault("missing_datasets", []).append(raw_key)
        else:
            result["raw_shape"] = _shape(raw_dataset)
            result["raw_spatial_shape"] = _raw_spatial_shape(result["raw_shape"])
            result["raw_channels"] = _raw_channel_count(result["raw_shape"])
            result["raw_dtype"] = str(raw_dataset.dtype)

        if not isinstance(label_dataset, h5py.Dataset):
            result.setdefault("missing_datasets", []).append(label_key)
            return result

        result["label_shape"] = _shape(label_dataset)
        result["label_dtype"] = str(label_dataset.dtype)
        result["shape_match"] = result["raw_spatial_shape"] == result["label_shape"]
        result.update(
            _analyze_labels(
                label_dataset,
                num_classes=num_classes,
                patch_shape=patch_shape,
                stride_shape=stride_shape,
                min_class_voxels=min_class_voxels,
            )
        )

    return result


def _format_list(values: list[Any] | None) -> str:
    if values is None:
        return "n/a"
    return "none" if len(values) == 0 else str(values)


def print_split_report(split: str, result: dict[str, Any], min_class_voxels: int) -> None:
    print(f"\n=== {split} ===")
    print(f"path: {result['path']}")

    if result.get("error") == "file_not_found":
        print("status: missing HDF5 file")
        return

    missing_datasets = result.get("missing_datasets", [])
    if missing_datasets:
        print(f"missing datasets: {missing_datasets}")

    print(
        f"raw:   shape={result.get('raw_shape')}, spatial={result.get('raw_spatial_shape')}, "
        f"channels={result.get('raw_channels')}, dtype={result.get('raw_dtype')}"
    )
    print(f"label: shape={result.get('label_shape')}, dtype={result.get('label_dtype')}")
    print(f"shape_match: {result.get('shape_match')}")

    if "unique_labels" not in result:
        return

    print(f"unique_labels: {_format_list(result['unique_labels'])}")
    print(f"invalid_labels: {_format_list(result['invalid_labels'])}")
    print(f"missing_classes: {_format_list(result['missing_classes'])}")

    print("\nclass distribution:")
    print("  class |       voxels |   fraction")
    print("  ------+--------------+-----------")
    for class_id, count in result["class_counts"].items():
        fraction = result["class_fractions"][class_id]
        print(f"  {int(class_id):5d} | {count:12d} | {fraction:9.6f}")

    coverage = result["patch_coverage"]
    print("\npatch coverage:")
    if coverage.get("note"):
        print(f"  note: {coverage['note']}")
    if coverage.get("skipped_reason"):
        print(f"  skipped: {coverage['skipped_reason']}")
    print(f"  total candidate patches: {coverage['total_patches']}")
    print(f"  class | >=1 voxel patches |   frac | >={min_class_voxels} voxel patches |   frac")
    print("  ------+-------------------+--------+-----------------------+--------")
    for class_id, stats in coverage["per_class"].items():
        print(
            f"  {int(class_id):5d} | "
            f"{stats['patches_with_class']:17d} | "
            f"{stats['fraction_with_class']:6.3f} | "
            f"{stats['patches_with_min_voxels']:21d} | "
            f"{stats['fraction_with_min_voxels']:6.3f}"
        )


def main() -> None:
    args = parse_args()
    h5_dir = Path(args.h5_dir)
    out_json = Path(args.out_json)
    patch_shape = tuple(args.patch_shape)
    stride_shape = tuple(args.stride_shape)

    if args.num_classes <= 0:
        raise ValueError(f"--num-classes must be positive, got {args.num_classes}")
    if args.min_class_voxels <= 0:
        raise ValueError(f"--min-class-voxels must be positive, got {args.min_class_voxels}")
    _require_positive("--patch-shape", patch_shape)
    _require_positive("--stride-shape", stride_shape)

    report: dict[str, Any] = {
        "config": {
            "h5_dir": str(h5_dir),
            "splits": list(args.splits),
            "raw_key": args.raw_key,
            "label_key": args.label_key,
            "num_classes": args.num_classes,
            "patch_shape": list(patch_shape),
            "stride_shape": list(stride_shape),
            "min_class_voxels": args.min_class_voxels,
        },
        "splits": {},
    }

    for split in args.splits:
        path = h5_dir / f"parihaka_{split}.h5"
        result = analyze_split(
            path,
            raw_key=args.raw_key,
            label_key=args.label_key,
            num_classes=args.num_classes,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            min_class_voxels=args.min_class_voxels,
        )
        report["splits"][split] = result
        print_split_report(split, result, args.min_class_voxels)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report: {out_json}")


if __name__ == "__main__":
    main()
