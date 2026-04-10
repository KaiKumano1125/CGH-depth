# CGH-Depth

A deep learning pipeline for Computer-Generated Holography (CGH) that predicts hologram **amplitude** and **phase** from RGB + depth images, using physics-aware frequency encoding and a bottleneck cross-attention mechanism.

---

## Project Goal

Given an RGB image and a depth map, predict the amplitude and phase of a hologram that can be physically reconstructed at multiple depths using the **Angular Spectrum Method (ASM)**.

---

## Method Overview

Three experiments are compared in this project:

| Experiment | Input channels | Cross-Attention | Description |
|---|---|---|---|
| Exp1 — Baseline | RGB + depth (4ch) | OFF | Standard UNet, no physics encoding |
| Exp2 — Concat | RGB + depth + cos + sin (6ch) | OFF | Physics channels mixed into main encoder |
| Exp3 — Cross-Attention | RGB + depth + cos + sin (6ch) | ON | Physics channels via separate encoder, injected at bottleneck |

The key idea of **Exp3**: the cos/sin frequency channels (derived from the ASM phase kernel `2π√(1/λ² - fx² - fy²) × z`) are processed by a dedicated lightweight encoder, then cross-attended at the UNet bottleneck. This keeps physics information pure until the highest semantic abstraction level.

### Lightweight Cross-Attention Architecture

```
Input (B, 6, 512, 512)
│
├── Main UNet Encoder (all 6 channels)
│   inc → down1 → down2 → down3 → down4
│   → (B, 1024, 32, 32)   ← bottleneck x5
│
├── FrequencyContextEncoder (cos + sin only)
│   AdaptiveAvgPool(32×32) → Conv1×1 → BN → ReLU
│   → (B, 1024, 32, 32)   ← freq_context
│
└── CrossAttentionBlock
    Pool: 32×32 → 8×8 (64 tokens)
    Project: 1024 → 256 (inner_dim)
    Cross-Attention (Q=x5, K/V=freq_context), 4 heads
    Project back: 256 → 1024
    Upsample: 8×8 → 32×32
    Residual: x5 = x5 + refined
│
└── UNet Decoder with skip connections → (B, 2, 512, 512)
    channel 0: amplitude
    channel 1: phase
```

---

## Folder Structure

```
configs/
  experiments/
    exp1_baseline.toml          ← Exp1: 4ch, no attention
    exp2_concat.toml            ← Exp2: 6ch, no attention
    exp3_cross_attention.toml   ← Exp3: 6ch, cross-attention (proposed)
    base.toml                   ← original baseline config
scripts/
  train_experiment.py           ← train a single experiment
  run_inference.py              ← run inference for a single experiment
  run_analysis.py               ← compare models, save plots and CSV
  run_comparison.py             ← train + compare all 3 experiments
  run_reconstruct_asm.py        ← standalone ASM reconstruction
src/cgh_depth/
  config.py                     ← TOML config loader → frozen dataclasses
  encoders.py                   ← physics-aware input encoder (cos/sin + RGB + depth)
  datasets.py                   ← KOREATECHHolographyDataset
  models.py                     ← SimpleUNet + FrequencyContextEncoder + CrossAttentionBlock
  training.py                   ← training loop with tqdm progress bars + TensorBoard
  checkpoints.py                ← checkpoint save/load, epoch inference from filename
  inference.py                  ← single-sample and batch prediction
  analysis.py                   ← ASM reconstruction, PSNR/SSIM evaluation, plots
dataset/
  KOREATECH-CGH-512-3.6Mu/     ← not tracked by git
    train/  validation/  test/
      img/  depth/  amp/  phs/
weight/
  v3/                           ← not tracked by git
    exp1_baseline_epoch_100.pth
    exp3_cross_attention_epoch_100.pth
results/
  analysis/
    batch/                      ← batch PSNR/SSIM CSV + plot
    samples/{id}/               ← per-sample comparison + hologram grid
```

---

## Quick Start

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

Check dataset loads correctly:

```bash
python scripts/train_experiment.py --config configs/experiments/exp1_baseline.toml --inspect-only
```

---

## Training

Train each experiment one by one:

```bash
# Exp1 — Baseline
python scripts/train_experiment.py --config configs/experiments/exp1_baseline.toml

# Exp2 — Concat
python scripts/train_experiment.py --config configs/experiments/exp2_concat.toml

# Exp3 — Cross-Attention
python scripts/train_experiment.py --config configs/experiments/exp3_cross_attention.toml
```

Training shows progress bars per batch and per epoch. Checkpoints are saved to `weight/v3/` every 10 epochs.

**Resume from checkpoint** — set in config:
```toml
resume_checkpoint = "weight/v3/exp1_baseline_epoch_50.pth"
```

---

## Inference

```bash
# Single sample
python scripts/run_inference.py --config configs/experiments/exp1_baseline.toml

# Full test set
python scripts/run_inference.py --config configs/experiments/exp1_baseline.toml --batch

# Custom RGB + depth pair
python scripts/run_inference.py \
  --config configs/experiments/exp3_cross_attention.toml \
  --rgb-path dataset/example/rgb.exr \
  --depth-path dataset/example/depth.exr
```

---

## Analysis & Comparison

**PSNR/SSIM comparison across depths (batch):**
```bash
python scripts/run_analysis.py \
  --config configs/experiments/exp1_baseline.toml \
  --config configs/experiments/exp3_cross_attention.toml \
  --batch
```

**Single sample — metrics + hologram grid:**
```bash
python scripts/run_analysis.py \
  --config configs/experiments/exp1_baseline.toml \
  --config configs/experiments/exp3_cross_attention.toml \
  --sample-id 5799 --holograms
```

**Control which depths appear in the hologram grid:**
```bash
  --display-depths-mm 5 10 15
```

Output is saved to:
```
results/analysis/
  batch/
    comparison_plots.png
    holography_comparison_results.csv
  samples/5799/
    comparison.png        ← PSNR/SSIM curves
    hologram_grid.png     ← visual reconstruction grid
```

The hologram grid layout:
```
              5.0 mm      10.0 mm     15.0 mm
Ground Truth  [image]     [image]     [image]
Exp1 Baseline [image]     [image]     [image]
Exp3 Cross... [image]     [image]     [image]
```

---

## Config Parameters

```toml
[encoder]
res           = 512        # image resolution
pitch         = 3.6e-6     # pixel pitch (m)
wavelength    = 638e-9     # light wavelength (m)
depth_range_m = 0.0203336  # max scene depth in meters (~20.3mm)
include_rgb       = true
include_depth     = true
include_freq_cos  = true   # ASM cos encoding
include_freq_sin  = true   # ASM sin encoding

[model]
name               = "simple_unet"
out_channels       = 2          # amplitude + phase
base_channels      = 64         # UNet width (bottleneck = 64×16 = 1024)
use_cross_attention = true      # Exp3 only

[train]
batch_size        = 4
learning_rate     = 1e-4
epochs            = 100
checkpoint_every  = 10
resume_checkpoint = ""          # set to resume from a checkpoint

[inference]
checkpoint          = "weight/v3/exp1_baseline_epoch_100.pth"
test_index          = "5799"
batch_output_subdir = "predictions"
prediction_prefix   = "pred_exp1"
```

---

## Evaluation Method

PSNR and SSIM are **not** computed on raw amplitude/phase. They are computed on the **ASM-reconstructed intensity image** at each depth:

```
predicted amp + phs
      ↓  ASM propagation at depth z
intensity image at z
      ↓  compare with ground truth intensity
PSNR / SSIM
```

This is the physically correct evaluation since hologram quality depends on how the light reconstructs at a given focal plane.
