from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import ExperimentConfig


def infer_start_epoch_from_checkpoint(path: Path) -> int:
    import re
    match = re.search(r"_epoch_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> int:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return infer_start_epoch_from_checkpoint(checkpoint_path)


def _latest_checkpoint(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    return max(candidates, key=infer_start_epoch_from_checkpoint)


def resolve_inference_checkpoint(config: ExperimentConfig) -> Path:
    if config.inference.checkpoint:
        checkpoint_path = config.resolve_path(config.inference.checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint_path}")

    weight_dir = config.weight_dir
    patterns = [
        f"*{config.experiment_name}*.pth",
        f"*{config.experiment_name.replace('_encoding', '')}*.pth",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        for path in weight_dir.glob(pattern):
            if path not in candidates:
                candidates.append(path)

    resolved = _latest_checkpoint(candidates)
    if resolved is None:
        raise FileNotFoundError(
            f"No checkpoint found for experiment '{config.experiment_name}' in {weight_dir}"
        )
    return resolved
