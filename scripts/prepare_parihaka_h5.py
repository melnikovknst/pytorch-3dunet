#!/usr/bin/env python3
"""Convert local Parihaka-3D NPZ volumes into spatial train/val/test HDF5 splits."""

from __future__ import annotations

import argparse
import ast
import json
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA = "../data/ampl3d/parihaka_data.npz"
DEFAULT_LABELS = "../data/ampl3d/parihaka_labels.npz"
FALLBACK_DATA = "../data/ampl_3d/parihaka_data.npz"
FALLBACK_LABELS = "../data/ampl_3d/parihaka_labels.npz"
DEFAULT_OUT_DIR = "outputs/h5"
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to parihaka_data.npz")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to parihaka_labels.npz")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for generated HDF5 files")
    parser.add_argument("--data-key", default=None, help="Explicit NPZ key for the amplitude volume")
    parser.add_argument("--label-key", default=None, help="Explicit NPZ key for the label volume")
    parser.add_argument("--split-axis", type=int, default=2, help="Spatial axis used for contiguous train/val/test split")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true", help="Inspect and print planned splits without writing HDF5")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def _read_npy_header(npz: zipfile.ZipFile, member_name: str) -> dict[str, Any]:
    with npz.open(member_name) as member:
        magic = member.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{member_name!r} in NPZ is not a valid NPY member")
        version = member.read(2)
        header_length_size = 2 if version[0] == 1 else 4
        header_length_format = "<H" if header_length_size == 2 else "<I"
        header_length = struct.unpack(header_length_format, member.read(header_length_size))[0]
        return ast.literal_eval(member.read(header_length).decode("latin1"))


def _array_infos(path: Path) -> list[dict[str, Any]]:
    infos = []
    with zipfile.ZipFile(path) as npz:
        for member_name in npz.namelist():
            if not member_name.endswith(".npy"):
                continue
            header = _read_npy_header(npz, member_name)
            dtype = np.dtype(header["descr"])
            shape = tuple(int(v) for v in header["shape"])
            key = Path(member_name).stem
            nbytes = int(np.prod(shape, dtype=np.int64) * dtype.itemsize)
            infos.append(
                {
                    "key": key,
                    "shape": shape,
                    "dtype": str(dtype),
                    "ndim": int(len(shape)),
                    "nbytes": nbytes,
                    "numeric": bool(np.issubdtype(dtype, np.number)),
                    "integer": bool(np.issubdtype(dtype, np.integer)),
                    "floating": bool(np.issubdtype(dtype, np.floating)),
                }
            )
    if not infos:
        raise ValueError(f"No NPY arrays found inside NPZ file: {path}")
    return infos


def _print_inventory(path: Path, infos: list[dict[str, Any]]) -> None:
    print(f"\n{path}")
    print(f"keys: {[info['key'] for info in infos]}")
    for info in infos:
        print(
            f"  - key={info['key']!r}, shape={info['shape']}, dtype={info['dtype']}, "
            f"memory={info['nbytes'] / 1024 ** 2:.2f} MiB"
        )


def _score_key(key: str, info: dict[str, Any], kind: str) -> int:
    lower = key.lower()
    score = 0
    if info["ndim"] == 3:
        score += 8
    if info["numeric"]:
        score += 2

    if kind == "data":
        hints = ("ampl", "amplitude", "data", "raw", "seismic", "volume")
        if info["floating"]:
            score += 2
    else:
        hints = ("label", "labels", "mask", "seg", "segmentation", "gt", "class", "facies")
        if info["integer"]:
            score += 3

    if any(hint in lower for hint in hints):
        score += 10
    if lower == "arr_0":
        score += 1
    return score


def _select_key(infos: list[dict[str, Any]], requested_key: str | None, kind: str) -> str:
    if requested_key is not None:
        available_keys = [info["key"] for info in infos]
        if requested_key not in available_keys:
            raise KeyError(f"Requested {kind} key {requested_key!r} not found. Available keys: {available_keys}")
        return requested_key

    numeric_infos = [info for info in infos if info["numeric"]]
    if len(numeric_infos) == 1:
        return numeric_infos[0]["key"]

    candidates = [info for info in numeric_infos if info["ndim"] == 3]
    if not candidates:
        raise ValueError(
            f"Could not find a numeric 3D {kind} array. Available keys/shapes: "
            f"{[(info['key'], info['shape'], info['dtype']) for info in infos]}"
        )

    scored = sorted(((_score_key(info["key"], info, kind), info) for info in candidates), key=lambda item: item[0])
    best_score, best_info = scored[-1]
    tied = [info for score, info in scored if score == best_score]
    if best_score <= 0 or len(tied) > 1:
        raise ValueError(
            f"Could not choose a unique {kind} key. Available keys/shapes: "
            f"{[(info['key'], info['shape'], info['dtype']) for info in infos]}. "
            f"Pass --{'data-key' if kind == 'data' else 'label-key'} explicitly."
        )
    return best_info["key"]


def _normalize_labels(labels: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    unique_before = np.unique(labels)
    unique_before_list = [int(v) for v in unique_before.tolist()]

    if unique_before.size == 0:
        raise ValueError("Label array is empty")

    if np.array_equal(unique_before, np.arange(1, 7)):
        labels = labels - 1
    elif unique_before.min() >= 1 and unique_before.max() <= 6:
        print("Labels are within 1..6; shifting them to 0..5.")
        labels = labels - 1
    elif unique_before.min() >= 0 and unique_before.max() <= 5:
        pass
    else:
        raise ValueError(
            f"Unsupported label values: {unique_before_list}. Expected classes 0..5 or 1..6 for CrossEntropyLoss."
        )

    labels = labels.astype(np.uint8, copy=False)
    unique_after_list = [int(v) for v in np.unique(labels).tolist()]
    return labels, unique_before_list, unique_after_list


def _split_ranges(axis_size: int, ratios: tuple[float, float, float]) -> dict[str, tuple[int, int]]:
    if axis_size <= 0:
        raise ValueError(f"Invalid split axis size: {axis_size}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"Ratios must be non-negative, got: {ratios}")
    ratio_sum = sum(ratios)
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.8f}")

    raw_counts = np.array(ratios, dtype=np.float64) * axis_size
    counts = np.floor(raw_counts).astype(int)
    for idx in np.argsort(-(raw_counts - counts))[: axis_size - int(counts.sum())]:
        counts[idx] += 1

    if axis_size >= len([r for r in ratios if r > 0]):
        for idx, ratio in enumerate(ratios):
            if ratio > 0 and counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[idx] += 1

    train_end = int(counts[0])
    val_end = int(counts[0] + counts[1])
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, axis_size),
    }


def _slice_volume(array: np.ndarray, axis: int, start: int, stop: int) -> np.ndarray:
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


def _write_h5(path: Path, raw: np.ndarray, labels: np.ndarray) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = tuple(min(dim, chunk) for dim, chunk in zip(raw.shape, (64, 128, 128), strict=True))
    with h5py.File(path, "w") as h5:
        h5.create_dataset("raw", data=raw, dtype="float32", chunks=chunks, compression="gzip")
        h5.create_dataset("label", data=labels, dtype="uint8", chunks=chunks, compression="gzip")


def main() -> None:
    args = parse_args()
    data_path = _resolve_path(args.data, DEFAULT_DATA, FALLBACK_DATA)
    labels_path = _resolve_path(args.labels, DEFAULT_LABELS, FALLBACK_LABELS)
    out_dir = Path(args.out_dir)

    _require_file(data_path)
    _require_file(labels_path)

    data_infos = _array_infos(data_path)
    label_infos = _array_infos(labels_path)
    data_key = _select_key(data_infos, args.data_key, "data")
    label_key = _select_key(label_infos, args.label_key, "label")
    _print_inventory(data_path, data_infos)
    _print_inventory(labels_path, label_infos)

    with np.load(data_path) as data_npz, np.load(labels_path) as label_npz:
        print(f"\nselected data key: {data_key!r}")
        print(f"selected label key: {label_key!r}")
        raw = np.asarray(data_npz[data_key], dtype=np.float32)
        labels = np.asarray(label_npz[label_key])

    if raw.ndim != 3 or labels.ndim != 3:
        raise ValueError(f"Expected 3D raw and label arrays, got raw.ndim={raw.ndim}, label.ndim={labels.ndim}")
    if raw.shape != labels.shape:
        raise ValueError(f"Raw and label shapes do not match: raw={raw.shape}, label={labels.shape}")
    if not 0 <= args.split_axis < raw.ndim:
        raise ValueError(f"--split-axis must be in [0, {raw.ndim - 1}], got {args.split_axis}")

    labels, unique_before, unique_after = _normalize_labels(labels)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    ranges = _split_ranges(raw.shape[args.split_axis], ratios)

    split_shapes = {}
    output_files = {}
    for split_name, (start, stop) in ranges.items():
        split_shape = list(raw.shape)
        split_shape[args.split_axis] = stop - start
        split_shapes[split_name] = split_shape
        output_files[split_name] = str(out_dir / f"parihaka_{split_name}.h5")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": {"data": str(data_path), "labels": str(labels_path)},
        "npz_keys": {"data": data_key, "label": label_key},
        "source_shapes": {"raw": list(raw.shape), "label": list(labels.shape)},
        "output_shapes": split_shapes,
        "dtype_raw": str(raw.dtype),
        "dtype_label": str(labels.dtype),
        "unique_labels_before": unique_before,
        "unique_labels_after": unique_after,
        "split_axis": args.split_axis,
        "split_axis_note": "Axis order is preserved; splits are contiguous spatial blocks along this axis.",
        "train_val_test_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "train_val_test_index_ranges": {name: list(value) for name, value in ranges.items()},
        "output_files": output_files,
        "parameters": vars(args),
    }

    print("\nunique labels before:", unique_before)
    print("unique labels after:", unique_after)
    print("split shapes:")
    for split_name in SPLIT_NAMES:
        print(f"  {split_name}: shape={split_shapes[split_name]}, range={ranges[split_name]}")

    if args.dry_run:
        print("\nDry run enabled; no HDF5 files or summary JSON were written.")
        print(json.dumps(summary, indent=2, default=_json_default))
        return

    for split_name, (start, stop) in ranges.items():
        raw_split = _slice_volume(raw, args.split_axis, start, stop)
        label_split = _slice_volume(labels, args.split_axis, start, stop)
        out_path = out_dir / f"parihaka_{split_name}.h5"
        print(f"writing {out_path}: raw={raw_split.shape}, label={label_split.shape}")
        _write_h5(out_path, raw_split, label_split)

    summary_path = out_dir / "parihaka_h5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
