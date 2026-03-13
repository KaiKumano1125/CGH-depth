from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cgh_depth.config import load_experiment_config
from cgh_depth.training import inspect_dataset, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "only_frequency.toml"),
    )
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.inspect_only:
        input_shape, target_shape = inspect_dataset(config)
        print(f"Input shape: {input_shape}")
        print(f"Target shape: {target_shape}")
        return

    run_training(config)


if __name__ == "__main__":
    main()
