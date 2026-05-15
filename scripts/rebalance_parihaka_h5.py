#!/usr/bin/env python3
"""Rebalance existing Parihaka HDF5 splits so configured patches fit."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


DEFAULT_H5_DIR = "outputs/h5"
SPLITS = ("train", "val", "test")
DATASETS = ("raw", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5-dir",
        default=DEFAULT_H5_DIR,
        help="Directory with parihaka_train/val/test.h5 files",
    )
    parser.add_argument(
        "--patch-shape",
        nargs=3,
        type=int,
        required=True,
        metavar=("D", "H", "W"),
        help="Minimum spatial shape required by the configured patch_shape",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=None,
        help="Spatial axis to rebalance. Defaults to summary split_axis, or 2 for the old Parihaka split.",
    )
    parser.add_argument("--chunk-depth", type=int, default=16, help="Number of slices copied at once")
    parser.add_argument("--dry-run", action="store_true", help="Print planned shapes without rewriting HDF5 files")
    return parser.parse_args()


def _dataset_axis(ndim: int, spatial_axis: int) -> int:
    axis = spatial_axis % 3
    return axis if ndim == 3 else axis + 1


def _slice(axis: int, start: int | None, stop: int | None, ndim: int) -> tuple[slice, ...]:
    slices = [slice(None)] * ndim
    slices[axis] = slice(start, stop)
    return tuple(slices)


def _copy_axis_range(
    src: h5py.Dataset,
    dst: h5py.Dataset,
    *,
    axis: int,
    src_start: int,
    dst_start: int,
    length: int,
    chunk_depth: int,
) -> None:
    for offset in range(0, length, chunk_depth):
        size = min(chunk_depth, length - offset)
        src_idx = _slice(axis, src_start + offset, src_start + offset + size, src.ndim)
        dst_idx = _slice(axis, dst_start + offset, dst_start + offset + size, dst.ndim)
        dst[dst_idx] = src[src_idx]


def _create_dataset(out_h5: h5py.File, name: str, shape: tuple[int, ...], dtype: Any) -> h5py.Dataset:
    chunk_template = (1, 64, 128, 128) if len(shape) == 4 else (64, 128, 128)
    chunks = tuple(min(dim, chunk) for dim, chunk in zip(shape, chunk_template, strict=True))
    return out_h5.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, compression="gzip")


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _ranges_from_lengths(lengths: dict[str, int]) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for split in SPLITS:
        stop = start + lengths[split]
        ranges[split] = (start, stop)
        start = stop
    return ranges


def _compute_new_lengths(old_lengths: dict[str, int], required_size: int) -> dict[str, int]:
    new_lengths = dict(old_lengths)

    for split in SPLITS[1:]:
        deficit = max(0, required_size - new_lengths[split])
        if deficit:
            new_lengths[split] += deficit
            new_lengths["train"] -= deficit

    if new_lengths["train"] < required_size:
        total_size = sum(old_lengths.values())
        raise ValueError(
            f"Cannot rebalance splits for patch axis size {required_size}: total axis size is {total_size}, "
            f"planned train size would be {new_lengths['train']}. "
            f"Use a smaller patch_shape or regenerate the splits with larger val/test ratios."
        )

    return new_lengths


def _validate_common_shapes(h5_files: dict[str, h5py.File], dataset_name: str, spatial_axis: int) -> int:
    reference = h5_files[SPLITS[0]][dataset_name]
    axis = _dataset_axis(reference.ndim, spatial_axis)
    reference_other = reference.shape[:axis] + reference.shape[axis + 1 :]

    for split in SPLITS:
        if dataset_name not in h5_files[split]:
            raise KeyError(f"Expected dataset {dataset_name!r} in parihaka_{split}.h5")

        dataset = h5_files[split][dataset_name]
        if dataset.ndim != reference.ndim:
            raise ValueError(f"{dataset_name}: {split} has ndim={dataset.ndim}, expected {reference.ndim}")

        dataset_axis = _dataset_axis(dataset.ndim, spatial_axis)
        other_shape = dataset.shape[:dataset_axis] + dataset.shape[dataset_axis + 1 :]
        if other_shape != reference_other:
            raise ValueError(
                f"{dataset_name}: non-rebalanced dimensions differ for {split}: "
                f"{other_shape} != {reference_other}"
            )

    return axis


def _copy_global_range(
    sources: list[tuple[h5py.Dataset, int, int]],
    dst: h5py.Dataset,
    *,
    axis: int,
    global_start: int,
    dst_start: int,
    length: int,
    chunk_depth: int,
) -> None:
    copied = 0
    cursor = global_start
    while copied < length:
        source_match = next((source for source in sources if source[1] <= cursor < source[2]), None)
        if source_match is None:
            raise ValueError(f"Cannot map global source position {cursor}")

        src_dataset, source_start, source_stop = source_match
        copy_length = min(length - copied, source_stop - cursor)
        _copy_axis_range(
            src_dataset,
            dst,
            axis=axis,
            src_start=cursor - source_start,
            dst_start=dst_start + copied,
            length=copy_length,
            chunk_depth=chunk_depth,
        )
        cursor += copy_length
        copied += copy_length


def _write_rebalanced_files(
    paths: dict[str, Path],
    tmp_paths: dict[str, Path],
    *,
    spatial_axis: int,
    old_lengths: dict[str, int],
    new_lengths: dict[str, int],
    chunk_depth: int,
) -> dict[str, list[int]]:
    output_shapes: dict[str, list[int]] = {}
    old_ranges = _ranges_from_lengths(old_lengths)
    new_ranges = _ranges_from_lengths(new_lengths)

    with (
        h5py.File(paths["train"], "r") as train_h5,
        h5py.File(paths["val"], "r") as val_h5,
        h5py.File(paths["test"], "r") as test_h5,
    ):
        h5_files = {"train": train_h5, "val": val_h5, "test": test_h5}
        dataset_axes = {
            dataset_name: _validate_common_shapes(h5_files, dataset_name, spatial_axis)
            for dataset_name in DATASETS
        }

        with (
            h5py.File(tmp_paths["train"], "w") as tmp_train_h5,
            h5py.File(tmp_paths["val"], "w") as tmp_val_h5,
            h5py.File(tmp_paths["test"], "w") as tmp_test_h5,
        ):
            tmp_files = {"train": tmp_train_h5, "val": tmp_val_h5, "test": tmp_test_h5}
            for split in SPLITS:
                _copy_attrs(h5_files[split].attrs, tmp_files[split].attrs)

            for dataset_name in DATASETS:
                axis = dataset_axes[dataset_name]
                reference = h5_files["train"][dataset_name]
                sources = [
                    (h5_files[split][dataset_name], old_ranges[split][0], old_ranges[split][1])
                    for split in SPLITS
                ]

                for split in SPLITS:
                    new_shape = list(reference.shape)
                    new_shape[axis] = new_lengths[split]
                    dst = _create_dataset(tmp_files[split], dataset_name, tuple(new_shape), reference.dtype)
                    _copy_attrs(reference.attrs, dst.attrs)

                    start, stop = new_ranges[split]
                    _copy_global_range(
                        sources,
                        dst,
                        axis=axis,
                        global_start=start,
                        dst_start=0,
                        length=stop - start,
                        chunk_depth=chunk_depth,
                    )

                    if dataset_name == "raw":
                        output_shapes[split] = new_shape[-3:]

    return output_shapes


def _update_summary(
    h5_dir: Path,
    output_shapes: dict[str, list[int]],
    spatial_axis: int,
    old_lengths: dict[str, int],
    new_lengths: dict[str, int],
) -> None:
    summary_path = h5_dir / "parihaka_h5_summary.json"
    if not summary_path.is_file():
        return

    backup_path = h5_dir / "parihaka_h5_summary.before_rebalance.json"
    if not backup_path.exists():
        shutil.copy2(summary_path, backup_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("rebalance_history", []).append(
        {
            "rebalanced_at": datetime.now(timezone.utc).isoformat(),
            "axis": spatial_axis,
            "old_lengths": old_lengths,
            "new_lengths": new_lengths,
        }
    )
    summary.setdefault("output_shapes", {}).update(output_shapes)

    summary["train_val_test_index_ranges"] = {
        split: list(value) for split, value in _ranges_from_lengths(new_lengths).items()
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _summary_axis(h5_dir: Path) -> int:
    summary_path = h5_dir / "parihaka_h5_summary.json"
    if not summary_path.is_file():
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary.get("split_axis", 2))


def main() -> None:
    args = parse_args()
    h5_dir = Path(args.h5_dir)
    paths = {split: h5_dir / f"parihaka_{split}.h5" for split in SPLITS}

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Expected existing train/val/test HDF5 files in {h5_dir}; missing: {missing}")

    patch_shape = tuple(args.patch_shape)
    spatial_axis = (args.axis if args.axis is not None else _summary_axis(h5_dir)) % 3
    required_size = patch_shape[spatial_axis]

    with (
        h5py.File(paths["train"], "r") as train_h5,
        h5py.File(paths["val"], "r") as val_h5,
        h5py.File(paths["test"], "r") as test_h5,
    ):
        h5_files = {"train": train_h5, "val": val_h5, "test": test_h5}
        axis = _validate_common_shapes(h5_files, "raw", spatial_axis)
        old_lengths = {split: int(h5_files[split]["raw"].shape[axis]) for split in SPLITS}

        for split in SPLITS:
            spatial_shape = h5_files[split]["raw"].shape[-3:]
            too_small_fixed_axes = [
                (idx, data_size, patch_size)
                for idx, (data_size, patch_size) in enumerate(zip(spatial_shape, patch_shape, strict=True))
                if idx != spatial_axis and data_size < patch_size
            ]
            if too_small_fixed_axes:
                raise ValueError(
                    f"Cannot rebalance {split}: non-split axes are smaller than patch_shape: "
                    f"{too_small_fixed_axes}"
                )

    new_lengths = _compute_new_lengths(old_lengths, required_size)

    print(f"Current split lengths on spatial axis {spatial_axis}: {old_lengths}")
    print(f"Required minimum on that axis from patch_shape {patch_shape}: {required_size}")

    if new_lengths == old_lengths:
        print("All splits are already large enough; no rebalance needed.")
        return

    print(f"Planned split lengths on spatial axis {spatial_axis}: {new_lengths}")
    if args.dry_run:
        print("Dry run enabled; HDF5 files were not rewritten.")
        return

    tmp_paths = {split: paths[split].with_suffix(".tmp_rebalanced.h5") for split in SPLITS}
    for tmp_path in tmp_paths.values():
        tmp_path.unlink(missing_ok=True)

    output_shapes = _write_rebalanced_files(
        paths,
        tmp_paths,
        spatial_axis=spatial_axis,
        old_lengths=old_lengths,
        new_lengths=new_lengths,
        chunk_depth=args.chunk_depth,
    )

    for split in SPLITS:
        tmp_paths[split].replace(paths[split])
    _update_summary(h5_dir, output_shapes, spatial_axis, old_lengths, new_lengths)

    for split in SPLITS:
        print(f"Rebalanced {split} shape: {output_shapes[split]}")


if __name__ == "__main__":
    main()
