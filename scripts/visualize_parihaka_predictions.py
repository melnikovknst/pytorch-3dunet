#!/usr/bin/env python3
"""Visualize Parihaka-3D test split predictions against expert labels."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fontconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


DEFAULT_H5 = "outputs/h5_parihaka/parihaka_test.h5"
DEFAULT_PRED_DIR = "outputs/predictions_parihaka"
DEFAULT_OUT_DIR = "outputs/visualizations_parihaka"
PREDICTION_KEY_PRIORITY = ("segmentation", "predictions", "prediction", "pred")
NUM_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5, help="HDF5 file with raw and label datasets")
    parser.add_argument(
        "--pred",
        "--prediction",
        dest="pred",
        default=None,
        help="Prediction HDF5 file or directory; auto-detected if omitted",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for PNGs and summary JSON")
    parser.add_argument("--num-slices", type=int, default=10, help="Number of evenly spaced slices to render")
    parser.add_argument("--indices", nargs="+", type=int, default=None, help="Explicit slice indices to render")
    parser.add_argument("--axis", type=int, default=1, help="Slice axis")
    parser.add_argument("--raw-channel", type=int, default=0, help="Raw channel to visualize when raw is [C,D,H,W]")
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


def _find_prediction_file(pred: str | None) -> Path:
    search_path = Path(pred) if pred else Path(DEFAULT_PRED_DIR)
    if search_path.is_file():
        return search_path
    if not search_path.exists():
        raise FileNotFoundError(f"Prediction path not found: {search_path}")
    if not search_path.is_dir():
        raise FileNotFoundError(f"Prediction path is neither a file nor a directory: {search_path}")

    candidates = sorted(
        list(search_path.glob("*parihaka_test*_predictions*.h5"))
        + list(search_path.glob("*parihaka_test*.h5"))
        + list(search_path.glob("*.h5"))
        + list(search_path.glob("*.hdf5"))
        + list(search_path.glob("*.hdf"))
    )
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    if not unique_candidates:
        raise FileNotFoundError(f"No HDF5 prediction files found in {search_path}")
    if len(unique_candidates) > 1:
        print("Multiple prediction files found; using:", unique_candidates[0])
    return unique_candidates[0]


def _collect_h5_datasets(h5: h5py.File) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    h5.visititems(visitor)
    return datasets


def _load_prediction(pred_path: Path, label_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    with h5py.File(pred_path, "r") as h5:
        datasets = _collect_h5_datasets(h5)
        if not datasets:
            raise ValueError(f"No datasets found in prediction file: {pred_path}")

        selected_key = None
        for preferred in PREDICTION_KEY_PRIORITY:
            for key in datasets:
                if key.split("/")[-1].lower() == preferred:
                    selected_key = key
                    break
            if selected_key is not None:
                break
        if selected_key is None:
            selected_key = sorted(datasets)[0]
            print(f"No standard prediction key found; using first dataset: {selected_key!r}")

        prediction = datasets[selected_key][...]

    if prediction.ndim == len(label_shape) + 1 and prediction.shape[0] == 1 and prediction.shape[1:] == label_shape:
        prediction = prediction[0]
    elif prediction.ndim == len(label_shape) + 1 and prediction.shape[1:] == label_shape:
        prediction = np.argmax(prediction, axis=0)
    elif prediction.ndim == len(label_shape) and prediction.shape == label_shape:
        pass
    else:
        raise ValueError(
            f"Unsupported prediction shape {prediction.shape}; expected {label_shape} or [C, *{label_shape}]"
        )

    return prediction.astype(np.uint8, copy=False), selected_key


def _slice_indices(length: int, num_slices: int) -> list[int]:
    if num_slices <= 0:
        raise ValueError("--num-slices must be positive")
    if length <= 0:
        raise ValueError("Cannot choose slices from an empty axis")
    if length == 1:
        return [0]

    start = min(max(int(np.floor(length * 0.05)), 0), length - 1)
    stop = min(max(int(np.ceil(length * 0.95)) - 1, start), length - 1)
    indices = np.rint(np.linspace(start, stop, min(num_slices, length))).astype(int).tolist()
    deduped = []
    for idx in indices:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def _validate_indices(indices: list[int], length: int, axis: int) -> list[int]:
    if not indices:
        raise ValueError("--indices must contain at least one value when provided")
    invalid = [index for index in indices if index < 0 or index >= length]
    if invalid:
        raise ValueError(f"--indices out of bounds for axis {axis} with length {length}: {invalid}")
    deduped = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    return deduped


def _take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis)


def _mask_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    colors = ["#111111", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    cmap = ListedColormap(colors, name="parihaka_classes")
    norm = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5, 1), cmap.N)
    return cmap, norm


def _error_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(["#f7f7f7", "#d62728"], name="parihaka_error")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    return cmap, norm


def _amplitude_limits(raw_slice: np.ndarray) -> tuple[float, float]:
    vmin, vmax = np.percentile(raw_slice, [1, 99])
    if np.isclose(vmin, vmax):
        vmin, vmax = float(np.min(raw_slice)), float(np.max(raw_slice))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _render_slice(
    raw_slice: np.ndarray,
    label_slice: np.ndarray,
    pred_slice: np.ndarray,
    axis: int,
    index: int,
    out_path: Path,
) -> None:
    mask_cmap, mask_norm = _mask_cmap()
    error_cmap, error_norm = _error_cmap()
    error_slice = pred_slice != label_slice
    vmin, vmax = _amplitude_limits(raw_slice)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    axes[0].imshow(raw_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Raw amplitude axis={axis} index={index}")
    axes[1].imshow(label_slice, cmap=mask_cmap, norm=mask_norm)
    axes[1].set_title("Expert Label")
    axes[2].imshow(pred_slice, cmap=mask_cmap, norm=mask_norm)
    axes[2].set_title("Predicted Mask")
    axes[3].imshow(error_slice.astype(np.uint8), cmap=error_cmap, norm=error_norm)
    axes[3].set_title("Error Map")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(axes[1].images[0], ax=axes[1], ticks=range(NUM_CLASSES), fraction=0.046, pad=0.04)
    fig.colorbar(axes[2].images[0], ax=axes[2], ticks=range(NUM_CLASSES), fraction=0.046, pad=0.04)
    fig.colorbar(axes[3].images[0], ax=axes[3], ticks=[0, 1], fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _render_grid(
    raw: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    indices: list[int],
    axis: int,
    out_path: Path,
) -> None:
    mask_cmap, mask_norm = _mask_cmap()
    error_cmap, error_norm = _error_cmap()
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 3.0 * len(indices)), constrained_layout=True)
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    mask_image = None
    error_image = None
    for row, index in enumerate(indices):
        raw_slice = _take_slice(raw, axis, index)
        label_slice = _take_slice(label, axis, index)
        pred_slice = _take_slice(pred, axis, index)
        error_slice = pred_slice != label_slice
        vmin, vmax = _amplitude_limits(raw_slice)

        axes[row, 0].imshow(raw_slice, cmap="gray", vmin=vmin, vmax=vmax)
        mask_image = axes[row, 1].imshow(label_slice, cmap=mask_cmap, norm=mask_norm)
        axes[row, 2].imshow(pred_slice, cmap=mask_cmap, norm=mask_norm)
        error_image = axes[row, 3].imshow(error_slice.astype(np.uint8), cmap=error_cmap, norm=error_norm)

        axes[row, 0].set_ylabel(f"{index}", rotation=0, labelpad=24, va="center")
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, title in enumerate(("Raw amplitude", "Expert Label", "Predicted Mask", "Error Map")):
        axes[0, col].set_title(title)

    if mask_image is not None:
        fig.colorbar(mask_image, ax=axes[:, 1:3], ticks=range(NUM_CLASSES), fraction=0.02, pad=0.01)
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[:, 3], ticks=[0, 1], fraction=0.02, pad=0.01)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _metrics(label: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float | None], float | None]:
    voxel_accuracy = float(np.mean(pred == label))
    per_class_iou: dict[str, float | None] = {}
    valid_ious = []
    for class_id in range(NUM_CLASSES):
        label_mask = label == class_id
        pred_mask = pred == class_id
        intersection = int(np.logical_and(label_mask, pred_mask).sum())
        union = int(np.logical_or(label_mask, pred_mask).sum())
        if union == 0:
            per_class_iou[str(class_id)] = None
        else:
            iou = intersection / union
            per_class_iou[str(class_id)] = float(iou)
            valid_ious.append(iou)
    mean_iou = float(np.mean(valid_ious)) if valid_ious else None
    return voxel_accuracy, per_class_iou, mean_iou


def _class_distribution(array: np.ndarray) -> dict[str, dict[str, float | int]]:
    total = int(array.size)
    distribution = {}
    for class_id in range(NUM_CLASSES):
        count = int(np.count_nonzero(array == class_id))
        distribution[str(class_id)] = {
            "count": count,
            "fraction": float(count / total) if total else 0.0,
        }
    return distribution


def _confusion_matrix(label: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    flat_index = NUM_CLASSES * label.astype(np.int64, copy=False).ravel() + pred.astype(np.int64, copy=False).ravel()
    matrix = np.bincount(flat_index, minlength=NUM_CLASSES * NUM_CLASSES)
    return matrix.reshape(NUM_CLASSES, NUM_CLASSES).astype(int).tolist()


def main() -> None:
    args = parse_args()
    h5_path = Path(args.h5)
    pred_path = _find_prediction_file(args.pred)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        raw = h5["raw"][...]
        label = h5["label"][...].astype(np.uint8, copy=False)

    if raw.ndim == 3:
        raw_for_display = raw
        raw_channels = 1
        if args.raw_channel != 0:
            raise ValueError(f"--raw-channel must be 0 for 3D raw, got {args.raw_channel}")
    elif raw.ndim == 4:
        raw_channels = int(raw.shape[0])
        if not 0 <= args.raw_channel < raw_channels:
            raise ValueError(f"--raw-channel must be in [0, {raw_channels - 1}], got {args.raw_channel}")
        raw_for_display = raw[args.raw_channel]
    else:
        raise ValueError(f"Unsupported raw shape {raw.shape}; expected [D,H,W] or [C,D,H,W]")

    if raw_for_display.shape != label.shape:
        raise ValueError(f"Raw spatial and label shapes differ: raw={raw_for_display.shape}, label={label.shape}")
    if not 0 <= args.axis < label.ndim:
        raise ValueError(f"--axis must be in [0, {label.ndim - 1}], got {args.axis}")

    pred, prediction_key = _load_prediction(pred_path, label.shape)
    if pred.shape != label.shape:
        raise ValueError(f"Prediction and label shapes differ: prediction={pred.shape}, label={label.shape}")

    indices = (
        _validate_indices(args.indices, label.shape[args.axis], args.axis)
        if args.indices is not None
        else _slice_indices(label.shape[args.axis], args.num_slices)
    )
    for out_index, slice_index in enumerate(indices):
        _render_slice(
            _take_slice(raw_for_display, args.axis, slice_index),
            _take_slice(label, args.axis, slice_index),
            _take_slice(pred, args.axis, slice_index),
            args.axis,
            slice_index,
            out_dir / f"slice_{out_index:03d}.png",
        )

    _render_grid(raw_for_display, label, pred, indices, args.axis, out_dir / "parihaka_10_slices_grid.png")
    voxel_accuracy, per_class_iou, mean_iou = _metrics(label, pred)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "h5_path": str(h5_path),
        "prediction_path": str(pred_path),
        "prediction_key": prediction_key,
        "shapes": {
            "raw": list(raw.shape),
            "raw_display": list(raw_for_display.shape),
            "label": list(label.shape),
            "prediction": list(pred.shape),
        },
        "raw_channels": raw_channels,
        "raw_channel": args.raw_channel,
        "slice_indices": indices,
        "axis": args.axis,
        "voxel_accuracy": voxel_accuracy,
        "per_class_iou": per_class_iou,
        "mean_iou": mean_iou,
        "label_distribution": _class_distribution(label),
        "prediction_distribution": _class_distribution(pred),
        "confusion_matrix_rows_label_cols_prediction": _confusion_matrix(label, pred),
        "num_classes": NUM_CLASSES,
    }
    summary_path = out_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote visualizations to {out_dir}")
    print(f"wrote {summary_path}")
    print(f"voxel_accuracy={voxel_accuracy:.6f}")
    print(f"mean_iou={mean_iou:.6f}" if mean_iou is not None else "mean_iou=null")
    print("per_class_iou:")
    for class_id, iou in per_class_iou.items():
        print(f"  class_{class_id}={iou:.6f}" if iou is not None else f"  class_{class_id}=null")


if __name__ == "__main__":
    main()
