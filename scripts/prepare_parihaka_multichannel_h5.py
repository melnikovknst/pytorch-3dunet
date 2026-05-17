#!/usr/bin/env python3
"""Prepare the main Parihaka HDF5 pipeline: full-volume attributes, axis=1 split, train-global normalization."""

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
DEFAULT_OUT_DIR = "outputs/h5_parihaka"
DEFAULT_PATCH_SHAPE = (64, 128, 128)
DEFAULT_LOCAL_STD_WINDOW = (5, 5, 5)
SPLIT_NAMES = ("train", "val", "test")
CHANNEL_NAMES = ("0_amplitude", "1_local_std", "2_horizontal_gradient_magnitude")
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to parihaka_data.npz")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to parihaka_labels.npz")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for generated HDF5 files")
    parser.add_argument("--data-key", default=None, help="Explicit NPZ key for the amplitude volume")
    parser.add_argument("--label-key", default=None, help="Explicit NPZ key for the label volume")
    parser.add_argument("--split-axis", type=int, default=1, help="Spatial axis for contiguous train/val/test split")
    parser.add_argument(
        "--patch-shape",
        nargs=3,
        type=int,
        default=list(DEFAULT_PATCH_SHAPE),
        metavar=("D", "H", "W"),
        help="Minimum spatial shape that every split must support",
    )
    parser.add_argument(
        "--local-std-window",
        nargs=3,
        type=int,
        default=list(DEFAULT_LOCAL_STD_WINDOW),
        metavar=("D", "H", "W"),
        help="Local window for channel 1 local standard deviation",
    )
    parser.add_argument("--train-ratio", type=float, default=0.67)
    parser.add_argument("--val-ratio", type=float, default=0.165)
    parser.add_argument("--test-ratio", type=float, default=0.165)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--class-weight-method", default="median_frequency", choices=("median_frequency",))
    parser.add_argument("--class-weight-clip-min", type=float, default=0.5)
    parser.add_argument("--class-weight-clip-max", type=float, default=6.0)
    parser.add_argument(
        "--update-train-config",
        default="configs/parihaka_train.yaml",
        help="YAML config whose loss.weight should be updated after class weights are computed; use '' to disable",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned outputs without writing HDF5")
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


def _require_positive(name: str, values: tuple[int, ...]) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive, got {values}")


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
            shape = tuple(int(value) for value in header["shape"])
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
    unique_before_list = [int(value) for value in unique_before.tolist()]
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
    unique_after_list = [int(value) for value in np.unique(labels).tolist()]
    return labels, unique_before_list, unique_after_list


def _split_ranges(
    axis_size: int,
    ratios: tuple[float, float, float],
    *,
    min_size: int = 1,
    adjust_to_min: bool = False,
) -> dict[str, tuple[int, int]]:
    if axis_size <= 0:
        raise ValueError(f"Invalid split axis size: {axis_size}")
    if min_size <= 0:
        raise ValueError(f"min_size must be positive, got {min_size}")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"Ratios must be non-negative, got: {ratios}")
    ratio_sum = sum(ratios)
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.8f}")

    active_indices = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if min_size * len(active_indices) > axis_size:
        raise ValueError(
            f"Cannot split axis of length {axis_size}: {len(active_indices)} non-empty splits require at least "
            f"{min_size * len(active_indices)} voxels to satisfy min split size {min_size}."
        )

    raw_counts = np.array(ratios, dtype=np.float64) * axis_size
    counts = np.floor(raw_counts).astype(int)
    for idx in np.argsort(-(raw_counts - counts))[: axis_size - int(counts.sum())]:
        counts[idx] += 1

    too_small = [SPLIT_NAMES[idx] for idx in active_indices if counts[idx] < min_size]
    if too_small and not adjust_to_min:
        raise ValueError(
            f"Split counts {dict(zip(SPLIT_NAMES, counts.tolist(), strict=True))} are too small for "
            f"min split size {min_size}: {too_small}. Change split ratios or patch_shape."
        )

    for idx in active_indices:
        deficit = min_size - int(counts[idx])
        if deficit <= 0:
            continue
        donor_order = sorted(active_indices, key=lambda donor_idx: int(counts[donor_idx] - min_size), reverse=True)
        for donor_idx in donor_order:
            if donor_idx == idx:
                continue
            surplus = int(counts[donor_idx] - min_size)
            if surplus <= 0:
                continue
            transfer = min(deficit, surplus)
            counts[donor_idx] -= transfer
            counts[idx] += transfer
            deficit -= transfer
            if deficit == 0:
                break
        if deficit > 0:
            raise ValueError(
                f"Cannot adjust split counts {dict(zip(SPLIT_NAMES, counts.tolist(), strict=True))} to satisfy "
                f"min split size {min_size}. Use a smaller patch_shape or different split ratios."
            )

    train_end = int(counts[0])
    val_end = int(counts[0] + counts[1])
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, axis_size)}


def _validate_split_shapes(
    source_shape: tuple[int, int, int],
    split_axis: int,
    ranges: dict[str, tuple[int, int]],
    patch_shape: tuple[int, int, int],
) -> dict[str, list[int]]:
    split_shapes = {}
    for split_name in SPLIT_NAMES:
        start, stop = ranges[split_name]
        split_shape = list(source_shape)
        split_shape[split_axis] = stop - start
        too_small = [
            f"axis {axis}: split={size}, patch={patch}"
            for axis, (size, patch) in enumerate(zip(split_shape, patch_shape, strict=True))
            if size < patch
        ]
        if too_small:
            raise ValueError(
                f"{split_name} split shape {tuple(split_shape)} is smaller than patch_shape {patch_shape}: {too_small}"
            )
        split_shapes[split_name] = split_shape
    return split_shapes


def _info_by_key(infos: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for info in infos:
        if info["key"] == key:
            return info
    raise KeyError(f"Key {key!r} not found in NPZ inventory")


def _validate_source_shapes(raw_shape: tuple[int, ...], label_shape: tuple[int, ...], split_axis: int) -> None:
    if len(raw_shape) != 3 or len(label_shape) != 3:
        raise ValueError(f"Expected 3D raw and label arrays, got raw={raw_shape}, label={label_shape}")
    if raw_shape != label_shape:
        raise ValueError(f"Raw and label shapes do not match: raw={raw_shape}, label={label_shape}")
    if not 0 <= split_axis < 3:
        raise ValueError(f"--split-axis must be in [0, 2], got {split_axis}")


def _slice_spatial(array: np.ndarray, axis: int, start: int, stop: int) -> np.ndarray:
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


def _slice_raw(raw: np.ndarray, split_axis: int, start: int, stop: int) -> np.ndarray:
    index = [slice(None)] * raw.ndim
    index[split_axis + 1] = slice(start, stop)
    return raw[tuple(index)]


def _compute_local_std(volume: np.ndarray, window: tuple[int, int, int]) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for local_std. Install project dependencies with `pip install -e .` "
            "or use the provided environment."
        ) from exc

    volume = volume.astype(np.float32, copy=False)
    mean = uniform_filter(volume, size=window, mode="reflect")
    mean_sq = uniform_filter(volume * volume, size=window, mode="reflect")
    var = mean_sq - mean * mean
    np.maximum(var, 0.0, out=var)
    np.sqrt(var, out=var)
    return var.astype(np.float32, copy=False)


def _compute_horizontal_gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    grad_h = np.gradient(volume, axis=1)
    grad_w = np.gradient(volume, axis=2)
    horizontal_gradient = grad_h.astype(np.float32, copy=False) ** 2
    horizontal_gradient += grad_w.astype(np.float32, copy=False) ** 2
    np.sqrt(horizontal_gradient, out=horizontal_gradient)
    return horizontal_gradient.astype(np.float32, copy=False)


def _build_full_multichannel_raw(amplitude: np.ndarray, local_std_window: tuple[int, int, int]) -> np.ndarray:
    amplitude = amplitude.astype(np.float32, copy=False)
    raw = np.empty((3,) + amplitude.shape, dtype=np.float32)
    raw[0] = amplitude
    raw[1] = _compute_local_std(amplitude, local_std_window)
    raw[2] = _compute_horizontal_gradient_magnitude(amplitude)
    return raw


def _normalize_channel_in_place(channel: np.ndarray, stats: dict[str, float]) -> None:
    np.clip(channel, stats["p01"], stats["p99"], out=channel)
    channel -= stats["mean_after_clip"]
    channel /= max(stats["std_after_clip"], EPS)


def _compute_train_stats(raw_train: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        channel = raw_train[channel_index]
        p01, p99 = np.percentile(channel, [1, 99])
        clipped = np.clip(channel, p01, p99)
        mean = float(np.mean(clipped, dtype=np.float64))
        std = float(np.std(clipped, dtype=np.float64))
        if std < EPS:
            print(f"WARNING: channel {channel_name} has near-zero std after clipping; using std=1.0")
            std = 1.0
        stats[channel_name] = {
            "p01": float(p01),
            "p99": float(p99),
            "mean_after_clip": mean,
            "std_after_clip": std,
        }
    return stats


def _apply_train_stats(raw: np.ndarray, stats: dict[str, dict[str, float]]) -> None:
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        _normalize_channel_in_place(raw[channel_index], stats[channel_name])


def _class_counts(label: np.ndarray, num_classes: int) -> np.ndarray:
    flat = label.reshape(-1)
    valid = flat[(flat >= 0) & (flat < num_classes)]
    return np.bincount(valid.astype(np.int64, copy=False), minlength=num_classes)[:num_classes]


def _median_frequency_weights(
    counts: np.ndarray,
    *,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute class weights from an empty train label split")
    fractions = counts.astype(np.float64) / total
    present = counts > 0
    if not np.any(present):
        raise ValueError("Cannot compute class weights: no classes are present in train label split")
    median_freq = float(np.median(fractions[present]))
    weights = np.full_like(fractions, fill_value=clip_max, dtype=np.float64)
    weights[present] = median_freq / fractions[present]
    weights = np.clip(weights, clip_min, clip_max)
    return fractions, weights


def _write_class_weights(
    out_dir: Path,
    *,
    counts: np.ndarray,
    fractions: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    ranges: dict[str, tuple[int, int]],
) -> None:
    weights_list = [float(f"{value:.6g}") for value in weights.tolist()]
    yaml_snippet = "loss:\n  name: WeightedCrossEntropyLoss\n  weight: [" + ", ".join(map(str, weights_list)) + "]\n"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.class_weight_method,
        "num_classes": args.num_classes,
        "clip_min": args.class_weight_clip_min,
        "clip_max": args.class_weight_clip_max,
        "split_axis": args.split_axis,
        "train_index_range": list(ranges["train"]),
        "class_counts": {str(class_id): int(counts[class_id]) for class_id in range(args.num_classes)},
        "class_fractions": {str(class_id): float(fractions[class_id]) for class_id in range(args.num_classes)},
        "weights": weights_list,
        "yaml_snippet": yaml_snippet,
    }
    json_path = out_dir / "parihaka_class_weights.json"
    yaml_path = out_dir / "parihaka_class_weights.yaml"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    yaml_path.write_text(yaml_snippet, encoding="utf-8")
    print(f"wrote class weights JSON: {json_path}")
    print(f"wrote class weights YAML: {yaml_path}")


def _update_train_config_weight(config_path: str, weights: np.ndarray) -> None:
    if not config_path:
        return
    path = Path(config_path)
    if not path.exists():
        print(f"WARNING: train config not found, skipping weight update: {path}")
        return

    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Cannot update invalid YAML config: {path}")
    loss_config = config.setdefault("loss", {})
    if not isinstance(loss_config, dict):
        raise ValueError(f"Cannot update config loss section because it is not a mapping: {path}")
    loss_config["name"] = "WeightedCrossEntropyLoss"
    loss_config["weight"] = [float(f"{value:.6g}") for value in weights.tolist()]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"updated loss weights in train config: {path}")


def _validate_split_arrays(raw: np.ndarray, label: np.ndarray, patch_shape: tuple[int, int, int], split_name: str) -> None:
    if raw.ndim != 4 or raw.shape[0] != 3:
        raise ValueError(f"{split_name}: expected raw shape [3,D,H,W], got {raw.shape}")
    if label.ndim != 3:
        raise ValueError(f"{split_name}: expected label shape [D,H,W], got {label.shape}")
    if raw.shape[1:] != label.shape:
        raise ValueError(f"{split_name}: raw spatial shape {raw.shape[1:]} does not match label {label.shape}")
    too_small = [
        f"axis {axis}: split={size}, patch={patch}"
        for axis, (size, patch) in enumerate(zip(label.shape, patch_shape, strict=True))
        if size < patch
    ]
    if too_small:
        raise ValueError(f"{split_name}: split shape {label.shape} is smaller than patch_shape {patch_shape}: {too_small}")
    unique = np.unique(label)
    if unique.size == 0 or int(unique.min()) < 0 or int(unique.max()) > 5:
        raise ValueError(f"{split_name}: labels must be in 0..5, got {unique.tolist()}")


def _write_h5(path: Path, raw: np.ndarray, label: np.ndarray) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    raw_chunks = (1,) + tuple(min(dim, chunk) for dim, chunk in zip(raw.shape[1:], (64, 128, 128), strict=True))
    label_chunks = tuple(min(dim, chunk) for dim, chunk in zip(label.shape, (64, 128, 128), strict=True))
    with h5py.File(path, "w") as h5:
        h5.create_dataset("raw", data=raw, dtype="float32", chunks=raw_chunks, compression="gzip")
        h5.create_dataset("label", data=label, dtype="uint8", chunks=label_chunks, compression="gzip")


def _planned_output_shapes(
    source_shape: tuple[int, int, int],
    ranges: dict[str, tuple[int, int]],
    split_axis: int,
) -> dict[str, dict[str, list[int]]]:
    output_shapes = {}
    for split_name in SPLIT_NAMES:
        start, stop = ranges[split_name]
        label_shape = list(source_shape)
        label_shape[split_axis] = stop - start
        output_shapes[split_name] = {"raw": [3] + label_shape, "label": label_shape}
    return output_shapes


def main() -> None:
    args = parse_args()
    data_path = _resolve_path(args.data, DEFAULT_DATA, FALLBACK_DATA)
    labels_path = _resolve_path(args.labels, DEFAULT_LABELS, FALLBACK_LABELS)
    out_dir = Path(args.out_dir)
    patch_shape = tuple(args.patch_shape)
    local_std_window = tuple(args.local_std_window)

    _require_file(data_path)
    _require_file(labels_path)
    _require_positive("--patch-shape", patch_shape)
    _require_positive("--local-std-window", local_std_window)
    if args.num_classes <= 0:
        raise ValueError(f"--num-classes must be positive, got {args.num_classes}")
    if args.class_weight_clip_min <= 0 or args.class_weight_clip_max <= 0:
        raise ValueError("Class weight clip bounds must be positive")
    if args.class_weight_clip_min > args.class_weight_clip_max:
        raise ValueError("--class-weight-clip-min cannot be greater than --class-weight-clip-max")

    data_infos = _array_infos(data_path)
    label_infos = _array_infos(labels_path)
    data_key = _select_key(data_infos, args.data_key, "data")
    label_key = _select_key(label_infos, args.label_key, "label")
    data_info = _info_by_key(data_infos, data_key)
    label_info = _info_by_key(label_infos, label_key)
    source_shape = tuple(int(value) for value in data_info["shape"])
    label_source_shape = tuple(int(value) for value in label_info["shape"])

    _print_inventory(data_path, data_infos)
    _print_inventory(labels_path, label_infos)
    print(f"\nselected data key: {data_key!r}")
    print(f"selected label key: {label_key!r}")

    _validate_source_shapes(source_shape, label_source_shape, args.split_axis)
    ranges = _split_ranges(
        source_shape[args.split_axis],
        (args.train_ratio, args.val_ratio, args.test_ratio),
        min_size=patch_shape[args.split_axis],
        adjust_to_min=False,
    )
    _validate_split_shapes(source_shape, args.split_axis, ranges, patch_shape)
    planned_shapes = _planned_output_shapes(source_shape, ranges, args.split_axis)
    output_files = {split: str(out_dir / f"parihaka_{split}.h5") for split in SPLIT_NAMES}

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": {"data": str(data_path), "labels": str(labels_path)},
        "npz_keys": {"data": data_key, "label": label_key},
        "source_shapes": {"raw": list(source_shape), "label": list(label_source_shape)},
        "output_shapes": planned_shapes,
        "channels": list(CHANNEL_NAMES),
        "local_std_window": list(local_std_window),
        "attribute_scope": "computed on the full amplitude volume before spatial split",
        "normalization_stats_json": str(out_dir / "parihaka_normalization_stats.json"),
        "class_weights_json": str(out_dir / "parihaka_class_weights.json"),
        "class_weights_yaml": str(out_dir / "parihaka_class_weights.yaml"),
        "split_axis": args.split_axis,
        "split_axis_note": "Axis order is preserved; splits are contiguous spatial blocks along this axis.",
        "patch_shape": list(patch_shape),
        "train_val_test_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "train_val_test_index_ranges": {name: list(value) for name, value in ranges.items()},
        "output_files": output_files,
        "parameters": vars(args),
    }

    print("planned split shapes:")
    for split_name in SPLIT_NAMES:
        print(
            f"  {split_name}: raw={planned_shapes[split_name]['raw']}, "
            f"label={planned_shapes[split_name]['label']}, range={ranges[split_name]}"
        )

    if args.dry_run:
        print("\nDry run enabled; no HDF5 files, stats JSON, class weight files, or summary JSON were written.")
        print(json.dumps(summary, indent=2, default=_json_default))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path) as data_npz, np.load(labels_path) as label_npz:
        amplitude = np.asarray(data_npz[data_key], dtype=np.float32)
        labels = np.asarray(label_npz[label_key])

    _validate_source_shapes(amplitude.shape, labels.shape, args.split_axis)
    labels, unique_before, unique_after = _normalize_labels(labels)
    summary["dtype_source_raw"] = str(amplitude.dtype)
    summary["dtype_label"] = str(labels.dtype)
    summary["unique_labels_before"] = unique_before
    summary["unique_labels_after"] = unique_after
    print("unique labels before:", unique_before)
    print("unique labels after:", unique_after)

    print("\ncomputing full-volume channels: amplitude, local_std, horizontal_gradient_magnitude")
    raw_full = _build_full_multichannel_raw(amplitude, local_std_window)
    del amplitude

    train_start, train_stop = ranges["train"]
    raw_train = _slice_raw(raw_full, args.split_axis, train_start, train_stop)
    label_train = _slice_spatial(labels, args.split_axis, train_start, train_stop)
    _validate_split_arrays(raw_train, label_train, patch_shape, "train")

    counts = _class_counts(label_train, args.num_classes)
    fractions, weights = _median_frequency_weights(
        counts,
        clip_min=args.class_weight_clip_min,
        clip_max=args.class_weight_clip_max,
    )
    _write_class_weights(out_dir, counts=counts, fractions=fractions, weights=weights, args=args, ranges=ranges)
    _update_train_config_weight(args.update_train_config, weights)

    normalization_stats = _compute_train_stats(raw_train)
    stats_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalization": "train_global_robust_per_channel",
        "source": str(data_path),
        "split_axis": args.split_axis,
        "train_index_range": list(ranges["train"]),
        "channels": normalization_stats,
    }
    stats_path = out_dir / "parihaka_normalization_stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote normalization stats: {stats_path}")

    for split_name in SPLIT_NAMES:
        start, stop = ranges[split_name]
        raw_split = _slice_raw(raw_full, args.split_axis, start, stop)
        label_split = _slice_spatial(labels, args.split_axis, start, stop)
        _validate_split_arrays(raw_split, label_split, patch_shape, split_name)
        _apply_train_stats(raw_split, normalization_stats)
        out_path = out_dir / f"parihaka_{split_name}.h5"
        print(f"writing {out_path}: raw={raw_split.shape}, label={label_split.shape}")
        _write_h5(out_path, raw_split, label_split)

    summary_path = out_dir / "parihaka_h5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
