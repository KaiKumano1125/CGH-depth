"""
run_comparison.py
=================
Orchestrates the three-experiment ablation study:

  Exp1 — RGB-D Baseline     : 4 channels, no physics encoding
  Exp2 — Concat             : 6 channels, physics mixed into main encoder
  Exp3 — Cross-Attention    : 6 channels, physics via separate encoder + bottleneck attention

Usage
-----
Train all three experiments sequentially, then compare:

    python scripts/run_comparison.py --mode all

Train only:
    python scripts/run_comparison.py --mode train

Compare pre-trained results (skip training):
    python scripts/run_comparison.py --mode compare

Compare a single test sample:
    python scripts/run_comparison.py --mode compare --sample-id 5799

Batch compare across entire test set:
    python scripts/run_comparison.py --mode compare --batch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cgh_depth.analysis import (
    evaluate_batch,
    evaluate_single_sample,
    plot_single_comparison,
    prediction_run_from_config,
    save_batch_summary,
)
from cgh_depth.config import load_experiment_config
from cgh_depth.inference import run_batch_inference
from cgh_depth.training import run_training

# ── Experiment definitions ────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = [
    ROOT / "configs" / "experiments" / "exp1_baseline.toml",
    ROOT / "configs" / "experiments" / "exp2_concat.toml",
    ROOT / "configs" / "experiments" / "exp3_cross_attention.toml",
]

DEFAULT_DEPTHS_MM = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "comparison"


# ── Train ─────────────────────────────────────────────────────────────────────

def train_all(configs: list) -> None:
    for config in configs:
        print(f"\n{'='*60}")
        print(f"  Training: {config.experiment_name}")
        print(f"{'='*60}")
        run_training(config)
        print(f"  Finished: {config.experiment_name}")


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_all(configs: list) -> None:
    for config in configs:
        print(f"\n  Running batch inference: {config.experiment_name}")
        output_dir = run_batch_inference(config)
        print(f"  Predictions saved to: {output_dir}")


# ── Compare ───────────────────────────────────────────────────────────────────

def compare_single(configs: list, sample_id: str, depths_m: list[float], output_dir: Path) -> None:
    runs = [prediction_run_from_config(c) for c in configs]
    data_root = configs[0].data_root

    comparison = evaluate_single_sample(data_root, runs, sample_id, depths_m)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sample_{sample_id}_comparison.png"
    plot_single_comparison(comparison, plot_path)
    print(f"\nSingle-sample comparison plot saved: {plot_path}")

    print(f"\n{'─'*50}")
    print(f"  Sample {sample_id} — average metrics across depths")
    print(f"{'─'*50}")
    for label, metrics in comparison.metrics.items():
        avg_psnr = sum(metrics["psnr"]) / len(metrics["psnr"])
        avg_ssim = sum(metrics["ssim"]) / len(metrics["ssim"])
        print(f"  {label:<30} PSNR={avg_psnr:.4f} dB   SSIM={avg_ssim:.4f}")


def compare_batch(configs: list, depths_m: list[float], output_dir: Path) -> None:
    runs = [prediction_run_from_config(c) for c in configs]
    data_root = configs[0].data_root

    dataframe = evaluate_batch(data_root, runs, depths_m)
    csv_path, plot_path = save_batch_summary(dataframe, output_dir)

    print(f"\nBatch comparison CSV  : {csv_path}")
    print(f"Batch comparison plot : {plot_path}")
    print(f"\n{dataframe.to_string(index=False)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Three-experiment ablation study runner.")
    parser.add_argument(
        "--mode",
        choices=["train", "compare", "all"],
        default="all",
        help=(
            "train  — train all three experiments\n"
            "compare — run inference + comparison on trained checkpoints\n"
            "all    — train then compare"
        ),
    )
    parser.add_argument("--sample-id", default=None, help="Single test sample ID for comparison.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run full batch comparison across the test set.",
    )
    parser.add_argument(
        "--depths-mm",
        nargs="+",
        type=float,
        default=DEFAULT_DEPTHS_MM,
        help="Reconstruction depths in millimeters.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output plots and CSVs.",
    )
    args = parser.parse_args()

    configs = [load_experiment_config(p) for p in EXPERIMENT_CONFIGS]
    depths_m = [d / 1000.0 for d in args.depths_mm]
    output_dir = Path(args.output_dir)

    if args.mode in ("train", "all"):
        train_all(configs)
        infer_all(configs)

    if args.mode in ("compare", "all"):
        if not args.sample_id and not args.batch:
            # Default: compare on the test_index from config + batch
            args.sample_id = configs[0].inference.test_index
            args.batch = True

        if args.sample_id:
            compare_single(configs, args.sample_id, depths_m, output_dir)

        if args.batch:
            compare_batch(configs, depths_m, output_dir)


if __name__ == "__main__":
    main()
