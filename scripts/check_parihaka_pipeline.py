#!/usr/bin/env python3
"""Check Parihaka HDF5/config compatibility for the multichannel axis=1 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


DEFAULT_H5_DIR = "outputs/h5_parihaka"
DEFAULT_TRAIN_CONFIG = "configs/parihaka_train.yaml"
DEFAULT_PREDICT_CONFIG = "configs/parihaka_predict.yaml"
SPLITS = ("train", "val", "test")
FORBIDDEN_TRANSFORMS = {"Standardize", "RandomFlip", "RandomRotate", "RandomRotate90", "ElasticDeformation"}
PREDICTION_KEY_PRIORITY = ("segmentation", "predictions", "prediction", "pred")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", default=DEFAULT_H5_DIR, help="Directory with parihaka_train/val/test.h5 files")
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG, help="Training YAML config")
    parser.add_argument("--predict-config", default=DEFAULT_PREDICT_CONFIG, help="Prediction YAML config")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--expected-channels", type=int, default=3)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config is empty or invalid: {path}")
    return data


def _spatial_shape(raw_shape: tuple[int, ...]) -> tuple[int, int, int] | None:
    if len(raw_shape) == 3:
        return tuple(int(value) for value in raw_shape)
    if len(raw_shape) == 4:
        return tuple(int(value) for value in raw_shape[1:])
    return None


def _channel_count(raw_shape: tuple[int, ...]) -> int | None:
    if len(raw_shape) == 3:
        return 1
    if len(raw_shape) == 4:
        return int(raw_shape[0])
    return None


def _label_values(label_dataset: h5py.Dataset) -> list[int | float | str]:
    values: set[int | float | str] = set()
    if label_dataset.ndim != 3:
        return [f"unsupported_label_ndim_{label_dataset.ndim}"]

    depth = int(label_dataset.shape[0])
    chunk_depth = max(1, min(32, depth))
    for start in range(0, depth, chunk_depth):
        chunk = label_dataset[start : start + chunk_depth]
        for value in np.unique(chunk).tolist():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool):
                values.add(int(value))
            elif isinstance(value, int):
                values.add(int(value))
            elif isinstance(value, float):
                values.add(int(value) if np.isfinite(value) and value.is_integer() else float(value))
            else:
                values.add(str(value))
    return sorted(values, key=lambda item: (str(type(item)), item))


def _valid_labels(values: list[int | float | str], num_classes: int) -> bool:
    for value in values:
        if not isinstance(value, int):
            return False
        if value < 0 or value >= num_classes:
            return False
    return True


def _inspect_h5(
    path: Path,
    *,
    raw_key: str,
    label_key: str,
    num_classes: int,
    expected_channels: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    result: dict[str, Any] = {"path": str(path)}
    if not path.is_file():
        return result, [f"Missing HDF5 split: {path}"]

    with h5py.File(path, "r") as h5:
        if raw_key not in h5:
            errors.append(f"{path}: missing raw dataset {raw_key!r}")
        if label_key not in h5:
            errors.append(f"{path}: missing label dataset {label_key!r}")
        if errors:
            return result, errors

        raw = h5[raw_key]
        label = h5[label_key]
        raw_shape = tuple(int(value) for value in raw.shape)
        label_shape = tuple(int(value) for value in label.shape)
        spatial_shape = _spatial_shape(raw_shape)
        channels = _channel_count(raw_shape)
        values = _label_values(label)

        result.update(
            {
                "raw_shape": raw_shape,
                "raw_spatial_shape": spatial_shape,
                "raw_channels": channels,
                "label_shape": label_shape,
                "labels": values,
            }
        )

        if spatial_shape is None:
            errors.append(f"{path}: raw shape must be [D,H,W] or [C,D,H,W], got {raw_shape}")
        if len(raw_shape) != 4:
            errors.append(f"{path}: expected multichannel raw [C,D,H,W], got {raw_shape}")
        elif channels != expected_channels:
            errors.append(f"{path}: expected C={expected_channels}, got C={channels}")
        if label.ndim != 3:
            errors.append(f"{path}: label shape must be [D,H,W], got {label_shape}")
        if spatial_shape is not None and spatial_shape != label_shape:
            errors.append(f"{path}: raw spatial shape {spatial_shape} does not match label shape {label_shape}")
        if not _valid_labels(values, num_classes):
            errors.append(f"{path}: labels must be integer class ids 0..{num_classes - 1}, got {values}")

    return result, errors


def _patch_fits(spatial_shape: tuple[int, int, int], patch_shape: tuple[int, int, int]) -> list[str]:
    return [
        f"axis {axis}: data={data_size}, patch={patch_size}"
        for axis, (data_size, patch_size) in enumerate(zip(spatial_shape, patch_shape, strict=True))
        if data_size < patch_size
    ]


def _transform_names(phase_config: dict[str, Any]) -> list[str]:
    names = []
    transformer = phase_config.get("transformer", {})
    for key in ("raw", "label"):
        for transform in transformer.get(key, []) or []:
            name = transform.get("name") if isinstance(transform, dict) else None
            if name:
                names.append(str(name))
    return names


def _collect_h5_datasets(h5: h5py.File) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    h5.visititems(visitor)
    return datasets


def _select_prediction_dataset(datasets: dict[str, h5py.Dataset], configured_key: str | None) -> str | None:
    if configured_key and configured_key in datasets:
        return configured_key
    for preferred in PREDICTION_KEY_PRIORITY:
        for key in datasets:
            if key.split("/")[-1].lower() == preferred:
                return key
    return sorted(datasets)[0] if datasets else None


def main() -> None:
    args = parse_args()
    h5_dir = Path(args.h5_dir)
    train_config_path = Path(args.train_config)
    predict_config_path = Path(args.predict_config)
    errors: list[str] = []
    warnings: list[str] = []

    if args.num_classes <= 0:
        errors.append(f"--num-classes must be positive, got {args.num_classes}")
    if args.expected_channels <= 0:
        errors.append(f"--expected-channels must be positive, got {args.expected_channels}")

    try:
        train_config = _load_yaml(train_config_path)
        predict_config = _load_yaml(predict_config_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    train_loaders = train_config.get("loaders", {})
    predict_loaders = predict_config.get("loaders", {})
    raw_key = train_loaders.get("raw_internal_path", "raw")
    label_key = train_loaders.get("label_internal_path", "label")
    predict_raw_key = predict_loaders.get("raw_internal_path", "raw")
    predict_label_key = predict_loaders.get("label_internal_path", "label")

    if predict_raw_key != raw_key:
        errors.append(f"raw_internal_path mismatch: train={raw_key!r}, predict={predict_raw_key!r}")
    if predict_label_key != label_key:
        errors.append(f"label_internal_path mismatch: train={label_key!r}, predict={predict_label_key!r}")

    split_info: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        info, split_errors = _inspect_h5(
            h5_dir / f"parihaka_{split}.h5",
            raw_key=raw_key,
            label_key=label_key,
            num_classes=args.num_classes,
            expected_channels=args.expected_channels,
        )
        split_info[split] = info
        errors.extend(split_errors)

    train_in_channels = train_config.get("model", {}).get("in_channels")
    predict_in_channels = predict_config.get("model", {}).get("in_channels")
    train_out_channels = train_config.get("model", {}).get("out_channels")
    predict_out_channels = predict_config.get("model", {}).get("out_channels")
    loss_config = train_config.get("loss", {})

    if train_in_channels != args.expected_channels:
        errors.append(f"train model.in_channels must be {args.expected_channels}, got {train_in_channels}")
    if predict_in_channels != train_in_channels:
        errors.append(f"predict model.in_channels {predict_in_channels} != train model.in_channels {train_in_channels}")
    if train_out_channels != args.num_classes:
        errors.append(f"train model.out_channels must be {args.num_classes}, got {train_out_channels}")
    if predict_out_channels != train_out_channels:
        errors.append(f"predict model.out_channels {predict_out_channels} != train model.out_channels {train_out_channels}")

    if not isinstance(loss_config, dict):
        errors.append("train config loss section must be a mapping")
    else:
        loss_name = loss_config.get("name")
        if loss_name not in ("WeightedCrossEntropyLoss", "CrossEntropyLoss"):
            errors.append(f"train config loss.name should be weighted CE, got {loss_name!r}")
        weights = loss_config.get("weight")
        if not isinstance(weights, list):
            errors.append("train config loss.weight must be a list")
        elif len(weights) != args.num_classes:
            errors.append(f"train config loss.weight length must be {args.num_classes}, got {len(weights)}")
        else:
            bad_weights = [value for value in weights if not isinstance(value, int | float) or float(value) <= 0]
            if bad_weights:
                errors.append(f"train config loss.weight must contain positive numbers, got bad values {bad_weights}")

    train_patch = tuple(train_loaders.get("train", {}).get("slice_builder", {}).get("patch_shape", ()))
    val_patch = tuple(train_loaders.get("val", {}).get("slice_builder", {}).get("patch_shape", ()))
    predict_patch = tuple(predict_loaders.get("test", {}).get("slice_builder", {}).get("patch_shape", ()))
    if len(train_patch) != 3:
        errors.append(f"train patch_shape must have 3 values, got {train_patch}")
    if val_patch != train_patch:
        errors.append(f"val patch_shape {val_patch} != train patch_shape {train_patch}")
    if predict_patch != train_patch:
        errors.append(f"predict patch_shape {predict_patch} != train patch_shape {train_patch}")

    for split, info in split_info.items():
        spatial_shape = info.get("raw_spatial_shape")
        if spatial_shape is not None and len(train_patch) == 3:
            too_small = _patch_fits(spatial_shape, train_patch)
            if too_small:
                errors.append(f"{split}: spatial shape {spatial_shape} is smaller than patch_shape {train_patch}: {too_small}")

    expected_paths = {
        "train": h5_dir / "parihaka_train.h5",
        "val": h5_dir / "parihaka_val.h5",
        "test": h5_dir / "parihaka_test.h5",
    }
    train_paths = [Path(path) for path in train_loaders.get("train", {}).get("file_paths", [])]
    val_paths = [Path(path) for path in train_loaders.get("val", {}).get("file_paths", [])]
    test_paths = [Path(path) for path in predict_loaders.get("test", {}).get("file_paths", [])]
    if expected_paths["train"] not in train_paths:
        errors.append(f"train config does not point to {expected_paths['train']}")
    if expected_paths["val"] not in val_paths:
        errors.append(f"train config val does not point to {expected_paths['val']}")
    if expected_paths["test"] not in test_paths:
        errors.append(f"predict config test does not point to {expected_paths['test']}")

    for phase_name, phase_config in (
        ("train", train_loaders.get("train", {})),
        ("val", train_loaders.get("val", {})),
        ("predict.test", predict_loaders.get("test", {})),
    ):
        forbidden = sorted(FORBIDDEN_TRANSFORMS.intersection(_transform_names(phase_config)))
        if forbidden:
            errors.append(f"{phase_name} uses forbidden transforms for this pipeline: {forbidden}")

    test_info = split_info.get("test", {})
    if test_info.get("raw_channels") is not None and test_info["raw_channels"] != predict_in_channels:
        errors.append(f"predict config in_channels {predict_in_channels} does not match test HDF5 C={test_info['raw_channels']}")

    if test_paths:
        test_path = test_paths[0]
        output_dir = Path(predict_loaders.get("output_dir", test_path.parent))
        prediction_path = output_dir / f"{test_path.stem}_predictions.h5"
        if prediction_path.exists():
            configured_prediction_key = predict_config.get("predictor", {}).get("output_dataset")
            with h5py.File(prediction_path, "r") as pred_h5:
                datasets = _collect_h5_datasets(pred_h5)
                selected_key = _select_prediction_dataset(datasets, configured_prediction_key)
                if selected_key is None:
                    errors.append(f"Prediction file has no datasets: {prediction_path}")
                else:
                    pred_shape = tuple(int(value) for value in datasets[selected_key].shape)
                    label_shape = test_info.get("label_shape")
                    if label_shape is not None:
                        if pred_shape == label_shape:
                            pass
                        elif len(pred_shape) == len(label_shape) + 1 and pred_shape[1:] == label_shape:
                            pass
                        else:
                            errors.append(
                                f"Prediction dataset {selected_key!r} shape {pred_shape} is incompatible with "
                                f"test label shape {label_shape}"
                            )
        else:
            warnings.append(f"Prediction file not present yet, skipped visualizer shape check: {prediction_path}")

    print("Parihaka pipeline compatibility report")
    for split in SPLITS:
        info = split_info.get(split, {})
        print(
            f"  {split}: raw={info.get('raw_shape')}, spatial={info.get('raw_spatial_shape')}, "
            f"label={info.get('label_shape')}, labels={info.get('labels')}"
        )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nOK: HDF5 files, configs, patch shapes, channels, labels, and transforms are compatible.")


if __name__ == "__main__":
    main()
