# CGH-depth

This repository now has a small reusable experiment scaffold so training and inference do not need to live entirely inside notebooks.

The main idea is:

- keep reusable logic in `src/cgh_depth`
- keep experiment settings in `configs`
- use `scripts` for training and inference entrypoints
- use notebooks mainly for analysis and visualization

## Quick Start

Use the project virtual environment. The system `python` on this machine does not include `pyexr`, but the repo virtualenv does.

```powershell
.\.venv\Scripts\python.exe
```

Check that the dataset and encoder wiring works:

```powershell
.\.venv\Scripts\python.exe scripts/train_experiment.py --inspect-only
```

Run training:

```powershell
.\.venv\Scripts\python.exe scripts/train_experiment.py --config configs/experiments/only_frequency.toml
```

Run single-image inference:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml
```

Run batch inference on the whole test set:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml --batch
```

## Folder Guide

```text
configs/
  base.toml
  experiments/
    only_frequency.toml
docs/
  project_structure.md
scripts/
  train_experiment.py
  run_inference.py
  run_analysis.py
src/
  cgh_depth/
    __init__.py
    analysis.py
    config.py
    checkpoints.py
    datasets.py
    encoders.py
    inference.py
    models.py
    training.py
notebooks/
  experiments/
```

## What Each New File Does

### `configs`

[`configs/base.toml`](c:/Users/Kai%20Kumano/workspace/CGH-depth/configs/base.toml)

- Template for shared settings.
- Holds default-style values for paths, encoder settings, model settings, training settings, and inference settings.
- Useful when you want to create more experiment configs later, such as `baseline.toml` or `proposed.toml`.

[`configs/experiments/only_frequency.toml`](c:/Users/Kai%20Kumano/workspace/CGH-depth/configs/experiments/only_frequency.toml)

- Main config for the current only-frequency experiment.
- Controls:
  - dataset path
  - output folders
  - encoder channels
  - model size
  - training hyperparameters
  - resume checkpoint
  - inference checkpoint
  - test sample id

### `scripts`

[`scripts/train_experiment.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/scripts/train_experiment.py)

- Command-line entrypoint for training.
- Loads a TOML config.
- Can either:
  - inspect the dataset shape with `--inspect-only`
  - run full training

Examples:

```powershell
.\.venv\Scripts\python.exe scripts/train_experiment.py --inspect-only
.\.venv\Scripts\python.exe scripts/train_experiment.py --config configs/experiments/only_frequency.toml
```

[`scripts/run_inference.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/scripts/run_inference.py)

- Command-line entrypoint for inference.
- Loads the config and trained checkpoint.
- By default it predicts one sample defined by `inference.test_index`.
- With `--batch`, it predicts the whole test set.

Examples:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml --batch
```

[`scripts/run_analysis.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/scripts/run_analysis.py)

- Command-line entrypoint for model comparison.
- Compares predictions from multiple experiment configs against ground-truth test holograms.
- Supports:
  - single-sample comparison
  - full batch comparison
- Saves plots and CSV summaries into `results/analysis` by default.

Examples:

```powershell
.\.venv\Scripts\python.exe scripts/run_analysis.py --config configs/experiments/base.toml --config configs/experiments/only_frequency.toml --sample-id 5799
.\.venv\Scripts\python.exe scripts/run_analysis.py --config configs/experiments/base.toml --config configs/experiments/only_frequency.toml --batch
```

### `src/cgh_depth`

[`src/cgh_depth/__init__.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/__init__.py)

- Small package entrypoint.
- Re-exports the config loader and config object.

[`src/cgh_depth/config.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/config.py)

- Loads TOML config files.
- Converts raw config values into structured dataclasses.
- Resolves relative paths like `dataset/...` into full project paths.

Use this when you want to load config inside a script or notebook:

```python
from cgh_depth.config import load_experiment_config

cfg = load_experiment_config("configs/experiments/only_frequency.toml")
```

[`src/cgh_depth/encoders.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/encoders.py)

- Contains `KOREATECHCGHEncoder`.
- Loads EXR files.
- Builds the frequency-based input features.
- Uses encoder config values like:
  - `res`
  - `pitch`
  - `wavelength`
  - `depth_range_m`
  - which channels to include

For the current config, the encoder creates 6 input channels:

- RGB: 3
- depth: 1
- frequency cosine: 1
- frequency sine: 1

[`src/cgh_depth/datasets.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/datasets.py)

- Contains `KOREATECHHolographyDataset`.
- Reads a split folder like `train` or `validation`.
- Uses the encoder to build the model input.
- Loads amplitude and phase as the target tensor.

[`src/cgh_depth/models.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/models.py)

- Contains:
  - `DoubleConv`
  - `SimpleUNet`
  - `build_model`
- `build_model(...)` uses the config to choose the model and input channel count.

[`src/cgh_depth/checkpoints.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/checkpoints.py)

- Handles checkpoint loading.
- Also tries to infer the start epoch from the checkpoint filename.

[`src/cgh_depth/training.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/training.py)

- Main training logic.
- Creates dataloaders.
- Builds the model.
- Runs train and validation loops.
- Logs to TensorBoard.
- Saves checkpoints every `checkpoint_every` epochs.

Main functions:

- `create_dataloaders(config)`
- `inspect_dataset(config)`
- `run_training(config)`

[`src/cgh_depth/inference.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/inference.py)

- Main inference logic.
- Loads the trained model.
- Runs either:
  - single-sample prediction
  - batch prediction for all test images

Main functions:

- `predict_single(config, sample_id=None)`
- `run_batch_inference(config)`

[`src/cgh_depth/analysis.py`](c:/Users/Kai%20Kumano/workspace/CGH-depth/src/cgh_depth/analysis.py)

- Reusable evaluation and comparison logic extracted from the old notebook analysis flow.
- Includes:
  - ASM reconstruction
  - EXR loading for predictions and ground truth
  - PSNR / SSIM comparison across depths
  - single-sample comparison plots
  - batch summary CSV and batch summary plots

Main functions:

- `prediction_run_from_config(config)`
- `evaluate_single_sample(...)`
- `evaluate_batch(...)`
- `plot_single_comparison(...)`
- `save_batch_summary(...)`

## How To Change Parameters

The main place to edit experiment settings is:

[`configs/experiments/only_frequency.toml`](c:/Users/Kai%20Kumano/workspace/CGH-depth/configs/experiments/only_frequency.toml)

### Common settings you will likely change

`[paths]`

- `data_root`: where the dataset is
- `weight_dir`: where checkpoints are saved
- `result_dir`: where inference outputs are saved
- `log_dir`: where TensorBoard logs are written

`[encoder]`

- `res`: input resolution
- `pitch`: pixel pitch
- `wavelength`: light wavelength
- `depth_range_m`: depth scaling used for `z_map`
- `include_rgb`
- `include_depth`
- `include_freq_cos`
- `include_freq_sin`

`[model]`

- `name`: currently `simple_unet`
- `out_channels`: currently 2 for amplitude and phase
- `base_channels`: width of the U-Net

`[train]`

- `batch_size`
- `learning_rate`
- `epochs`
- `shuffle`
- `num_workers`
- `checkpoint_every`
- `resume_checkpoint`

`[inference]`

- `checkpoint`: model weights to load
- `test_index`: sample id used for single-image inference
- `batch_output_subdir`: folder name under `result_dir` for batch outputs
- `prediction_prefix`: file prefix for saved EXRs

## Typical Workflow

### 1. Inspect shapes first

This is the safest quick test after changing encoder settings.

```powershell
.\.venv\Scripts\python.exe scripts/train_experiment.py --inspect-only
```

Expected current result:

```text
Input shape: torch.Size([8, 6, 512, 512])
Target shape: torch.Size([8, 2, 512, 512])
```

### 2. Train

```powershell
.\.venv\Scripts\python.exe scripts/train_experiment.py --config configs/experiments/only_frequency.toml
```

Training will:

- load the train and validation splits
- resume from `resume_checkpoint` if it exists
- save new checkpoints into `weight/v2`
- log TensorBoard files into `logs`

### 3. Run inference

Single sample:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml
```

Batch:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml --batch
```

### 4. Compare models

Single sample comparison:

```powershell
.\.venv\Scripts\python.exe scripts/run_analysis.py --config configs/experiments/base.toml --config configs/experiments/only_frequency.toml --sample-id 5799
```

This saves a plot like:

```text
results/analysis/single_5799_comparison.png
```

Batch comparison over the whole test set:

```powershell
.\.venv\Scripts\python.exe scripts/run_analysis.py --config configs/experiments/base.toml --config configs/experiments/only_frequency.toml --batch
```

This saves:

- `results/analysis/holography_comparison_results.csv`
- `results/analysis/comparison_plots.png`

Qualitative comparison for an arbitrary RGB/depth EXR pair:

```powershell
.\.venv\Scripts\python.exe scripts/run_analysis.py --config configs/experiments/base.toml --config configs/experiments/only_frequency.toml --rgb-path dataset/example/honney_rgb.exr --depth-path dataset/example/honney_dep.exr --depths-mm 5 10 15
```

This saves a side-by-side reconstruction plot such as:

- `results/analysis/example/example_honney_rgb_comparison.png`

## Notebook Usage

You can now keep notebooks thin and use them mainly for checking outputs, plotting, and experiments.

Example:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd().parents[1] / "src"))

from cgh_depth.config import load_experiment_config
from cgh_depth.training import inspect_dataset
from cgh_depth.inference import predict_single

cfg = load_experiment_config("configs/experiments/only_frequency.toml")
inspect_dataset(cfg)
amp, phs = predict_single(cfg)
```

## Recommended Next Step

The current notebook [`notebooks/experiments/proposed-feature-encoding-only-frequency-encoding.ipynb`](c:/Users/Kai%20Kumano/workspace/CGH-depth/notebooks/experiments/proposed-feature-encoding-only-frequency-encoding.ipynb) still contains old inline code. A good next cleanup would be to replace most of that notebook with:

- config loading
- one or two calls into `cgh_depth`
- analysis and visualization cells only

That would make the notebook much easier to maintain.
