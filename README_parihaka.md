# Parihaka 3D Facies Pipeline

## Multichannel axis=1 pipeline

This pipeline is the current recommended Parihaka setup. It changes the split axis from the old axis=2 baseline to
axis=1, writes multichannel raw tensors, and stores already normalized HDF5 volumes.

### One-job Slurm run

To run the whole multichannel sequence in one Slurm job:

```bash
sbatch sbatch/parihaka_multichannel_pipeline.sbatch
```

The job runs prepare if any multichannel HDF5 split is missing, then analyze, compatibility check, train, predict, and
visualization for axes 0/1/2. Logs follow the `unet_ampl.sbatch` style and are written to:

```text
outputs/slurm_logs/parihaka_multichannel_pipeline/<job_id>.out
outputs/slurm_logs/parihaka_multichannel_pipeline/<job_id>.err
```

Useful overrides:

```bash
FORCE_PREPARE=1 sbatch sbatch/parihaka_multichannel_pipeline.sbatch
RUN_PREDICT_AFTER_TRAIN=0 sbatch sbatch/parihaka_multichannel_pipeline.sbatch
RUN_VISUALIZATION_AFTER_PREDICT=0 sbatch sbatch/parihaka_multichannel_pipeline.sbatch
RESUME_CHECKPOINT=outputs/checkpoints_multichannel/last_checkpoint.pytorch sbatch sbatch/parihaka_multichannel_pipeline.sbatch
```

### 1. Prepare HDF5

```bash
python scripts/prepare_parihaka_multichannel_h5.py \
  --split-axis 1 \
  --out-dir outputs/h5_axis1_multichannel \
  --patch-shape 64 128 128
```

The script writes:

- `outputs/h5_axis1_multichannel/parihaka_train.h5`
- `outputs/h5_axis1_multichannel/parihaka_val.h5`
- `outputs/h5_axis1_multichannel/parihaka_test.h5`
- `outputs/diagnostics/parihaka_multichannel_raw_stats.json`

The HDF5 layout is:

- `raw`: `[C, D, H, W]`, where `C=3`
- `label`: `[D, H, W]`

Channels:

- `0_amplitude`: original seismic amplitude
- `1_coherence`: lightweight local semblance/coherence-like attribute from a small uniform-filter window
- `2_gradient`: 3D gradient magnitude from `np.gradient`

Normalization is train-global per channel. The prepare script computes train-only `p01`, `p99`,
`mean_after_clip`, and `std_after_clip`, then applies those same statistics to train/val/test. Do not add
`Standardize` back into the train or predict transforms for this pipeline.

### 2. Analyze HDF5/statistics

```bash
python scripts/analyze_parihaka_h5.py \
  --h5-dir outputs/h5_axis1_multichannel \
  --patch-shape 64 128 128 \
  --stride-shape 48 96 96
```

### 3. Check pipeline compatibility

```bash
python scripts/check_parihaka_pipeline.py \
  --h5-dir outputs/h5_axis1_multichannel \
  --train-config configs/parihaka_train_multichannel.yaml \
  --predict-config configs/parihaka_predict_multichannel.yaml
```

This checks HDF5 keys, shapes, channels, label range, config `in_channels`, patch fit, predict/test consistency,
and forbidden transforms such as `Standardize`, `RandomFlip`, rotations, and elastic deformation.

### 4. Train

```bash
sbatch sbatch/parihaka_train_multichannel.sbatch
```

Direct command equivalent:

```bash
train3dunet --config configs/parihaka_train_multichannel.yaml
```

### 5. Predict

```bash
sbatch sbatch/parihaka_predict_multichannel.sbatch
```

Direct command equivalent:

```bash
predict3dunet --config configs/parihaka_predict_multichannel.yaml
```

### 6. Visualize

Axis 0:

```bash
python scripts/visualize_parihaka_predictions.py \
  --h5 outputs/h5_axis1_multichannel/parihaka_test.h5 \
  --prediction outputs/predictions_multichannel/parihaka_test_predictions.h5 \
  --raw-channel 0 \
  --axis 0 \
  --indices 50 151 251 352 452 \
  --out-dir outputs/visualizations_multichannel/axis0
```

Axis 1:

```bash
python scripts/visualize_parihaka_predictions.py \
  --h5 outputs/h5_axis1_multichannel/parihaka_test.h5 \
  --prediction outputs/predictions_multichannel/parihaka_test_predictions.h5 \
  --raw-channel 0 \
  --axis 1 \
  --indices 10 40 70 100 130 \
  --out-dir outputs/visualizations_multichannel/axis1
```

Axis 2:

```bash
python scripts/visualize_parihaka_predictions.py \
  --h5 outputs/h5_axis1_multichannel/parihaka_test.h5 \
  --prediction outputs/predictions_multichannel/parihaka_test_predictions.h5 \
  --raw-channel 0 \
  --axis 2 \
  --indices 50 150 250 350 450 \
  --out-dir outputs/visualizations_multichannel/axis2
```

## Notes

The split is a contiguous spatial blocked split along axis=1. This is meant to test whether the old axis=2 baseline was
overusing positional/depth priors and producing horizontal bands instead of learning seismic geometry.

Spatial augmentations are intentionally disabled in the new configs. The raw and label transforms are only `ToTensor`.

Because axis=1 changes the spatial holdout, test class distribution can differ from the old baseline. Inspect visual
geometry for the major classes, especially 1 and 3, instead of relying only on mIoU.
