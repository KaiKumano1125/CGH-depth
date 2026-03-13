from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cgh_depth.config import load_experiment_config
from cgh_depth.inference import predict_single, run_batch_inference


def main() -> None:
    from cgh_depth.encoders import _load_pyexr

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "only_frequency.toml"),
    )
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.batch:
        output_dir = run_batch_inference(config)
        print(f"Batch predictions written to: {output_dir}")
        return

    pyexr = _load_pyexr()
    amp, phs = predict_single(config)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.inference.prediction_prefix
    sample = config.inference.test_index
    amp_path = config.result_dir / f"{prefix}_{sample}_amp.exr"
    phs_path = config.result_dir / f"{prefix}_{sample}_phs.exr"
    pyexr.write(str(amp_path), amp)
    pyexr.write(str(phs_path), phs)
    print(f"Saved single-sample prediction: {amp_path}")
    print(f"Saved single-sample prediction: {phs_path}")


if __name__ == "__main__":
    main()
