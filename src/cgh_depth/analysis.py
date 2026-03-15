from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .encoders import _load_pyexr
from .inference import predict_from_paths


@dataclass(frozen=True)
class PredictionRun:
    label: str
    data_root: Path
    primary_prediction_dir: Path
    fallback_prediction_dir: Path
    prefix: str


@dataclass
class SingleComparisonResult:
    sample_id: str
    depths_m: list[float]
    metrics: dict[str, dict[str, list[float]]]
    visuals: dict[str, list[np.ndarray]]


@dataclass
class InputPairComparisonResult:
    input_name: str
    depths_m: list[float]
    visuals: dict[str, list[np.ndarray]]


def prediction_run_from_config(config: ExperimentConfig, label: str | None = None) -> PredictionRun:
    return PredictionRun(
        label=label or config.experiment_name,
        data_root=config.data_root,
        primary_prediction_dir=config.result_dir / config.inference.batch_output_subdir,
        fallback_prediction_dir=config.result_dir,
        prefix=config.inference.prediction_prefix,
    )


def _analysis_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_exr_tensor(path: Path, device: torch.device) -> torch.Tensor:
    pyexr = _load_pyexr()
    data = pyexr.open(str(path)).get().astype(np.float32)
    if data.ndim == 3:
        data = data[:, :, 0]
    return torch.from_numpy(data).float().to(device)


def _normalize_image(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().numpy()
    image_min = float(image_np.min())
    image_max = float(image_np.max())
    return (image_np - image_min) / (image_max - image_min + 1e-8)


def _phase_to_radians(phs: torch.Tensor, phase_normalized: bool) -> torch.Tensor:
    if phase_normalized:
        return phs * (2.0 * np.pi)
    return phs


def reconstruct_asm(
    amp: torch.Tensor,
    phs: torch.Tensor,
    z: float,
    pitch: float = 3.6e-6,
    wavelength: float = 638e-9,
    phase_normalized: bool = True,
) -> torch.Tensor:
    res_y, res_x = amp.shape
    phase_radians = _phase_to_radians(phs, phase_normalized)
    u_hologram = torch.complex(amp * torch.cos(phase_radians), amp * torch.sin(phase_radians))
    u_freq = torch.fft.fftshift(torch.fft.fft2(u_hologram))

    fy = torch.fft.fftfreq(res_y, d=pitch, device=amp.device)
    fx = torch.fft.fftfreq(res_x, d=pitch, device=amp.device)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    fyy = torch.fft.fftshift(fyy)
    fxx = torch.fft.fftshift(fxx)

    k = 2.0 * np.pi / wavelength
    term = 1 - (wavelength * fxx) ** 2 - (wavelength * fyy) ** 2
    kernel = torch.exp(1j * k * z * torch.sqrt(torch.clamp(term, min=0)))
    u_reconstructed = torch.fft.ifft2(torch.fft.ifftshift(u_freq * kernel))
    return torch.abs(u_reconstructed) ** 2


def _load_ground_truth_pair(data_root: Path, sample_id: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gt_amp = _load_exr_tensor(data_root / "test" / "amp" / f"{sample_id}.exr", device)
    gt_phs = _load_exr_tensor(data_root / "test" / "phs" / f"{sample_id}.exr", device)
    return gt_amp, gt_phs


def _prediction_pair_candidates(run: PredictionRun, sample_id: str) -> list[tuple[Path, Path]]:
    candidate_dirs: list[Path] = []
    for directory in [
        run.primary_prediction_dir,
        run.fallback_prediction_dir,
        run.fallback_prediction_dir / "test",
        run.fallback_prediction_dir / "predictions",
    ]:
        if directory not in candidate_dirs:
            candidate_dirs.append(directory)

    return [
        (
            directory / f"{run.prefix}_{sample_id}_amp.exr",
            directory / f"{run.prefix}_{sample_id}_phs.exr",
        )
        for directory in candidate_dirs
    ]


def _load_prediction_pair(run: PredictionRun, sample_id: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    for amp_path, phs_path in _prediction_pair_candidates(run, sample_id):
        if amp_path.exists() and phs_path.exists():
            return _load_exr_tensor(amp_path, device), _load_exr_tensor(phs_path, device)
    raise FileNotFoundError(
        f"Prediction files for sample {sample_id} not found for '{run.label}'. "
        f"Checked: {run.primary_prediction_dir} and {run.fallback_prediction_dir}"
    )


def evaluate_single_sample(
    data_root: Path,
    runs: Iterable[PredictionRun],
    sample_id: str,
    depths_m: Iterable[float],
) -> SingleComparisonResult:
    device = _analysis_device()
    gt_amp, gt_phs = _load_ground_truth_pair(data_root, sample_id, device)
    depths = list(depths_m)
    run_list = list(runs)

    metrics = {run.label: {"psnr": [], "ssim": []} for run in run_list}
    visuals: dict[str, list[np.ndarray]] = {"GroundTruth": []}
    visuals.update({run.label: [] for run in run_list})

    prediction_pairs = {
        run.label: _load_prediction_pair(run, sample_id, device)
        for run in run_list
    }

    for z in depths:
        gt_recon = reconstruct_asm(gt_amp, gt_phs, z)
        gt_norm = _normalize_image(gt_recon)
        visuals["GroundTruth"].append(gt_norm)

        for run in run_list:
            pred_amp, pred_phs = prediction_pairs[run.label]
            pred_recon = reconstruct_asm(pred_amp, pred_phs, z)
            pred_norm = _normalize_image(pred_recon)
            visuals[run.label].append(pred_norm)
            metrics[run.label]["psnr"].append(psnr_metric(gt_norm, pred_norm, data_range=1.0))
            metrics[run.label]["ssim"].append(ssim_metric(gt_norm, pred_norm, data_range=1.0))

    return SingleComparisonResult(
        sample_id=sample_id,
        depths_m=depths,
        metrics=metrics,
        visuals=visuals,
    )


def plot_single_comparison(
    comparison: SingleComparisonResult,
    output_path: Path | None = None,
) -> plt.Figure:
    depth_labels = [z * 1000.0 for z in comparison.depths_m]
    display_indices = sorted({0, len(comparison.depths_m) // 2, len(comparison.depths_m) - 1})
    row_labels = list(comparison.visuals.keys())

    fig = plt.figure(figsize=(5 * len(display_indices), 3.5 * (len(row_labels) + 2)))
    gs = fig.add_gridspec(len(row_labels) + 2, len(display_indices))

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, depth_idx in enumerate(display_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(comparison.visuals[row_label][depth_idx], cmap="gray")
            ax.set_title(f"{row_label} @ {depth_labels[depth_idx]:.1f} mm")
            ax.axis("off")

    ax_psnr = fig.add_subplot(gs[len(row_labels), :])
    ax_ssim = fig.add_subplot(gs[len(row_labels) + 1, :])
    styles = ["o-", "s-", "x-", "d-", "^-", "v-"]
    for style, (label, metrics) in zip(styles, comparison.metrics.items()):
        ax_psnr.plot(depth_labels, metrics["psnr"], style, linewidth=2, label=label)
        ax_ssim.plot(depth_labels, metrics["ssim"], style, linewidth=2, label=label)

    ax_psnr.set_title(f"PSNR by Depth for Sample {comparison.sample_id}")
    ax_psnr.set_xlabel("Depth (mm)")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.legend()

    ax_ssim.set_title(f"SSIM by Depth for Sample {comparison.sample_id}")
    ax_ssim.set_xlabel("Depth (mm)")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.grid(True, alpha=0.3)
    ax_ssim.legend()

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def compare_input_pair(
    configs: Iterable[ExperimentConfig],
    img_path: Path,
    depth_path: Path,
    depths_m: Iterable[float],
) -> InputPairComparisonResult:
    device = _analysis_device()
    depths = list(depths_m)
    visuals: dict[str, list[np.ndarray]] = {}

    for config in configs:
        amp_pred, phs_pred = predict_from_paths(config, img_path, depth_path)
        amp_tensor = torch.from_numpy(amp_pred).float().to(device)
        phs_tensor = torch.from_numpy(phs_pred).float().to(device)
        visuals[config.experiment_name] = [
            _normalize_image(reconstruct_asm(amp_tensor, phs_tensor, z))
            for z in depths
        ]

    return InputPairComparisonResult(
        input_name=img_path.stem,
        depths_m=depths,
        visuals=visuals,
    )


def plot_input_pair_comparison(
    comparison: InputPairComparisonResult,
    output_path: Path | None = None,
) -> plt.Figure:
    depth_labels = [z * 1000.0 for z in comparison.depths_m]
    row_labels = list(comparison.visuals.keys())
    fig = plt.figure(figsize=(4.5 * len(depth_labels), 3.8 * len(row_labels)))
    gs = fig.add_gridspec(len(row_labels), len(depth_labels))

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, depth_mm in enumerate(depth_labels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(comparison.visuals[row_label][col_idx], cmap="gray")
            ax.set_title(f"{row_label} @ {depth_mm:.1f} mm")
            ax.axis("off")

    fig.suptitle(f"Qualitative Comparison for {comparison.input_name}", fontsize=16)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def evaluate_batch(
    data_root: Path,
    runs: Iterable[PredictionRun],
    depths_m: Iterable[float],
) -> pd.DataFrame:
    device = _analysis_device()
    run_list = list(runs)
    depths = list(depths_m)
    sample_ids = sorted(path.stem for path in (data_root / "test" / "amp").glob("*.exr"))

    stats = {
        run.label: {depth: {"psnr": [], "ssim": []} for depth in depths}
        for run in run_list
    }

    for sample_id in tqdm(sample_ids, desc="Analysis batch"):
        try:
            gt_amp, gt_phs = _load_ground_truth_pair(data_root, sample_id, device)
            prediction_pairs = {
                run.label: _load_prediction_pair(run, sample_id, device)
                for run in run_list
            }

            for z in depths:
                gt_norm = _normalize_image(reconstruct_asm(gt_amp, gt_phs, z))
                for run in run_list:
                    pred_amp, pred_phs = prediction_pairs[run.label]
                    pred_norm = _normalize_image(reconstruct_asm(pred_amp, pred_phs, z))
                    stats[run.label][z]["psnr"].append(psnr_metric(gt_norm, pred_norm, data_range=1.0))
                    stats[run.label][z]["ssim"].append(ssim_metric(gt_norm, pred_norm, data_range=1.0))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except FileNotFoundError as exc:
            print(f"Skipping {sample_id}: {exc}")

    rows: list[dict[str, float]] = []
    for z in depths:
        row: dict[str, float] = {"z_mm": z * 1000.0}
        for run in run_list:
            psnr_vals = stats[run.label][z]["psnr"]
            ssim_vals = stats[run.label][z]["ssim"]
            row[f"{run.label}_PSNR_Mean"] = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
            row[f"{run.label}_PSNR_Std"] = float(np.std(psnr_vals)) if psnr_vals else float("nan")
            row[f"{run.label}_SSIM_Mean"] = float(np.mean(ssim_vals)) if ssim_vals else float("nan")
            row[f"{run.label}_SSIM_Std"] = float(np.std(ssim_vals)) if ssim_vals else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def plot_batch_summary(dataframe: pd.DataFrame, labels: Iterable[str], output_path: Path | None = None) -> plt.Figure:
    labels = list(labels)
    d_mm = dataframe["z_mm"].tolist()

    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(18, 7))
    styles = ["o", "s", "x", "d", "^", "v"]

    for style, label in zip(styles, labels):
        ax_psnr.errorbar(
            d_mm,
            dataframe[f"{label}_PSNR_Mean"],
            yerr=dataframe[f"{label}_PSNR_Std"],
            fmt=f"{style}-",
            linewidth=2,
            capsize=4,
            label=label,
        )
        ax_ssim.errorbar(
            d_mm,
            dataframe[f"{label}_SSIM_Mean"],
            yerr=dataframe[f"{label}_SSIM_Std"],
            fmt=f"{style}-",
            linewidth=2,
            capsize=4,
            label=label,
        )

    ax_psnr.set_title("Reconstruction Quality: PSNR")
    ax_psnr.set_xlabel("Reconstruction Distance (mm)")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.legend()

    ax_ssim.set_title("Reconstruction Quality: SSIM")
    ax_ssim.set_xlabel("Reconstruction Distance (mm)")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.grid(True, alpha=0.3)
    ax_ssim.legend()

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def save_batch_summary(
    dataframe: pd.DataFrame,
    output_dir: Path,
    csv_name: str = "holography_comparison_results.csv",
    plot_name: str = "comparison_plots.png",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    plot_path = output_dir / plot_name
    dataframe.to_csv(csv_path, index=False)
    labels = sorted(
        {
            column[: -len("_PSNR_Mean")]
            for column in dataframe.columns
            if column.endswith("_PSNR_Mean")
        }
    )
    plot_batch_summary(dataframe, labels, plot_path)
    return csv_path, plot_path
