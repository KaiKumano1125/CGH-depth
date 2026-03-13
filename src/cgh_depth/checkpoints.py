from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def infer_start_epoch_from_checkpoint(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> int:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return infer_start_epoch_from_checkpoint(checkpoint_path)
