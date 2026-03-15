from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cgh_depth.analysis import (
    compare_input_pair,
    evaluate_batch,
    evaluate_single_sample,
    plot_input_pair_comparison,
    plot_single_comparison,
    prediction_run_from_config,
    save_batch_summary,
)
from cgh_depth.config import load_experiment_config


def _default_output_dir() -> Path:
    return ROOT / "results" / "analysis"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Experiment config to compare. Pass this argument multiple times.",
    )
    parser.add_argument("--sample-id", help="Run single-sample comparison for one test index.")
    parser.add_argument("--batch", action="store_true", help="Run full batch comparison across the test set.")
    parser.add_argument("--rgb-path", help="Arbitrary RGB EXR path for qualitative model comparison.")
    parser.add_argument("--depth-path", help="Arbitrary depth EXR path for qualitative model comparison.")
    parser.add_argument(
        "--depths-mm",
        nargs="+",
        type=float,
        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help="Depths in millimeters used for reconstruction comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory for CSV and plots.",
    )
    args = parser.parse_args()

    if bool(args.rgb_path) != bool(args.depth_path):
        parser.error("Use --rgb-path and --depth-path together.")

    modes_selected = sum(
        [
            bool(args.sample_id),
            bool(args.batch),
            bool(args.rgb_path and args.depth_path),
        ]
    )
    if modes_selected != 1:
        parser.error("Choose exactly one mode: --sample-id, --batch, or --rgb-path with --depth-path.")

    configs = [load_experiment_config(path) for path in args.config]
    data_root = configs[0].data_root
    runs = [prediction_run_from_config(config) for config in configs]
    depths_m = [depth / 1000.0 for depth in args.depths_mm]
    output_dir = Path(args.output_dir)

    if args.rgb_path and args.depth_path:
        comparison = compare_input_pair(
            configs,
            Path(args.rgb_path),
            Path(args.depth_path),
            depths_m,
        )
        plot_path = output_dir / f"example_{comparison.input_name}_comparison.png"
        plot_input_pair_comparison(comparison, plot_path)
        print(f"Saved arbitrary-input comparison plot: {plot_path}")
        return

    if args.sample_id:
        comparison = evaluate_single_sample(data_root, runs, args.sample_id, depths_m)
        plot_path = output_dir / f"{args.sample_id}_comparison.png"
        plot_single_comparison(comparison, plot_path)
        print(f"Saved single-sample comparison plot: {plot_path}")

        for label, metrics in comparison.metrics.items():
            avg_psnr = sum(metrics["psnr"]) / len(metrics["psnr"])
            avg_ssim = sum(metrics["ssim"]) / len(metrics["ssim"])
            print(f"{label}: avg PSNR={avg_psnr:.4f} dB, avg SSIM={avg_ssim:.4f}")

    if args.batch:
        dataframe = evaluate_batch(data_root, runs, depths_m)
        csv_path, plot_path = save_batch_summary(dataframe, output_dir)
        print(f"Saved batch metrics CSV: {csv_path}")
        print(f"Saved batch comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
