#!/usr/bin/env python3
"""Prepare Parihaka-3D multichannel HDF5 splits: amplitude, coherence-like attribute, gradient magnitude."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from prepare_parihaka_h5 import (
    DEFAULT_DATA,
    DEFAULT_LABELS,
    FALLBACK_DATA,
    FALLBACK_LABELS,
    SPLIT_NAMES,
    _array_infos,
    _json_default,
    _normalize_labels,
    _print_inventory,
    _require_file,
    _require_positive,
    _resolve_path,
    _select_key,
    _slice_volume,
    _split_ranges,
    _validate_split_shapes,
)


DEFAULT_OUT_DIR = "outputs/h5_axis1_multichannel"
DEFAULT_STATS_JSON = "outputs/diagnostics/parihaka_multichannel_raw_stats.json"
DEFAULT_PATCH_SHAPE = (64, 128, 128)
DEFAULT_COHERENCE_WINDOW = (3, 3, 3)
CHANNEL_NAMES = ("0_amplitude", "1_coherence", "2_gradient")
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to parihaka_data.npz")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to parihaka_labels.npz")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for generated HDF5 files")
    parser.add_argument("--stats-json", default=DEFAULT_STATS_JSON, help="Path for train-global normalization stats")
    parser.add_argument("--data-key", default=None, help="Explicit NPZ key for the amplitude volume")
    parser.add_argument("--label-key", default=None, help="Explicit NPZ key for the label volume")
    parser.add_argument("--split-axis", type=int, default=1, help="Spatial axis used for contiguous train/val/test split")
    parser.add_argument(
        "--patch-shape",
        nargs=3,
        type=int,
        default=list(DEFAULT_PATCH_SHAPE),
        metavar=("D", "H", "W"),
        help="Minimum spatial shape that every split must support",
    )
    parser.add_argument(
        "--coherence-window",
        nargs=3,
        type=int,
        default=list(DEFAULT_COHERENCE_WINDOW),
        metavar=("D", "H", "W"),
        help="Local window for the lightweight semblance/coherence-like attribute",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned splits without writing HDF5")
    return parser.parse_args()


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


def _compute_coherence(volume: np.ndarray, window: tuple[int, int, int]) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for the lightweight coherence-like attribute. "
            "Install project dependencies with `pip install -e .` or use the provided environment."
        ) from exc

    volume = volume.astype(np.float32, copy=False)
    local_mean = uniform_filter(volume, size=window, mode="reflect")
    local_energy = uniform_filter(volume * volume, size=window, mode="reflect")
    coherence = (local_mean * local_mean) / np.maximum(local_energy, EPS)
    return np.clip(coherence, 0.0, 1.0).astype(np.float32, copy=False)


def _compute_gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    gradients = np.gradient(volume.astype(np.float32, copy=False))
    grad_mag = np.zeros_like(volume, dtype=np.float32)
    for component in gradients:
        grad_mag += component.astype(np.float32, copy=False) ** 2
    np.sqrt(grad_mag, out=grad_mag)
    return grad_mag


def _build_multichannel_raw(amplitude: np.ndarray, coherence_window: tuple[int, int, int]) -> np.ndarray:
    amplitude = amplitude.astype(np.float32, copy=False)
    raw = np.empty((3,) + amplitude.shape, dtype=np.float32)
    raw[0] = amplitude
    raw[1] = _compute_coherence(amplitude, coherence_window)
    raw[2] = _compute_gradient_magnitude(amplitude)
    return raw


def _normalize_channel_in_place(channel: np.ndarray, stats: dict[str, float]) -> None:
    np.clip(channel, stats["p01"], stats["p99"], out=channel)
    channel -= stats["mean_after_clip"]
    channel /= max(stats["std_after_clip"], EPS)


def _compute_train_stats_and_normalize(raw: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        channel = raw[channel_index]
        p01, p99 = np.percentile(channel, [1, 99])
        np.clip(channel, p01, p99, out=channel)
        mean = float(np.mean(channel, dtype=np.float64))
        std = float(np.std(channel, dtype=np.float64))
        if std < EPS:
            print(f"WARNING: channel {channel_name} has near-zero std after clipping; using std=1.0")
            std = 1.0
        stats[channel_name] = {
            "p01": float(p01),
            "p99": float(p99),
            "mean_after_clip": mean,
            "std_after_clip": std,
        }
        _normalize_channel_in_place(channel, stats[channel_name])
    return stats


def _apply_train_stats(raw: np.ndarray, stats: dict[str, dict[str, float]]) -> None:
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        _normalize_channel_in_place(raw[channel_index], stats[channel_name])


def _validate_split_arrays(raw: np.ndarray, label: np.ndarray, patch_shape: tuple[int, int, int], split_name: str) -> None:
    if raw.ndim != 4 or raw.shape[0] != 3:
        raise ValueError(f"{split_name}: expected raw shape [3, D, H, W], got {raw.shape}")
    if label.ndim != 3:
        raise ValueError(f"{split_name}: expected label shape [D, H, W], got {label.shape}")
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
    stats_json = Path(args.stats_json)
    patch_shape = tuple(args.patch_shape)
    coherence_window = tuple(args.coherence_window)

    _require_file(data_path)
    _require_file(labels_path)
    _require_positive("--patch-shape", patch_shape)
    _require_positive("--coherence-window", coherence_window)

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
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    ranges = _split_ranges(source_shape[args.split_axis], ratios, min_size=patch_shape[args.split_axis])
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
        "coherence_window": list(coherence_window),
        "attribute_scope": "computed independently inside each contiguous split",
        "normalization": "robust train-global per channel: train p01/p99 clip, train mean/std after clip",
        "normalization_stats_json": str(stats_json),
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
        print("\nDry run enabled; no HDF5 files, stats JSON, or summary JSON were written.")
        print(json.dumps(summary, indent=2, default=_json_default))
        return

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

    train_stats: dict[str, dict[str, float]] | None = None
    for split_name in SPLIT_NAMES:
        start, stop = ranges[split_name]
        amplitude_split = _slice_volume(amplitude, args.split_axis, start, stop)
        label_split = _slice_volume(labels, args.split_axis, start, stop)
        print(f"\ncomputing channels for {split_name}: amplitude={amplitude_split.shape}, label={label_split.shape}")
        raw_split = _build_multichannel_raw(amplitude_split, coherence_window)
        _validate_split_arrays(raw_split, label_split, patch_shape, split_name)

        if split_name == "train":
            train_stats = _compute_train_stats_and_normalize(raw_split)
            stats_payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_train_h5": str(out_dir / "parihaka_train.h5"),
                "split_axis": args.split_axis,
                "train_index_range": list(ranges["train"]),
                "channels": train_stats,
            }
            stats_json.parent.mkdir(parents=True, exist_ok=True)
            stats_json.write_text(json.dumps(stats_payload, indent=2, default=_json_default), encoding="utf-8")
            print(f"wrote normalization stats: {stats_json}")
        else:
            if train_stats is None:
                raise RuntimeError("Internal error: train statistics are unavailable before val/test normalization")
            _apply_train_stats(raw_split, train_stats)

        out_path = out_dir / f"parihaka_{split_name}.h5"
        print(f"writing {out_path}: raw={raw_split.shape}, label={label_split.shape}")
        _write_h5(out_path, raw_split, label_split)

    summary_path = out_dir / "parihaka_h5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
