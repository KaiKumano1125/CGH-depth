from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .checkpoints import load_model_weights
from .config import ExperimentConfig
from .encoders import KOREATECHCGHEncoder, _load_pyexr
from .models import build_model


def _load_ready_model(config: ExperimentConfig) -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, config.encoder).to(device)
    checkpoint_path = config.resolve_path(config.inference.checkpoint)
    load_model_weights(model, checkpoint_path, device)
    model.eval()
    return model, device


def predict_single(config: ExperimentConfig, sample_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    model, device = _load_ready_model(config)
    encoder = KOREATECHCGHEncoder(config.encoder)
    sample = sample_id or config.inference.test_index

    img_path = config.data_root / "test" / "img" / f"{sample}.exr"
    depth_path = config.data_root / "test" / "depth" / f"{sample}.exr"
    x_input = encoder.encode(img_path, depth_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x_input).squeeze(0).cpu().numpy()

    return output[0], output[1]


def run_batch_inference(config: ExperimentConfig) -> Path:
    pyexr = _load_pyexr()
    model, device = _load_ready_model(config)
    encoder = KOREATECHCGHEncoder(config.encoder)

    test_img_dir = config.data_root / "test" / "img"
    test_depth_dir = config.data_root / "test" / "depth"
    output_dir = config.result_dir / config.inference.batch_output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = sorted(path.stem for path in test_img_dir.glob("*.exr"))
    for sample_id in tqdm(indices, desc="Batch inference"):
        x_input = encoder.encode(
            test_img_dir / f"{sample_id}.exr",
            test_depth_dir / f"{sample_id}.exr",
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x_input).squeeze(0).cpu().numpy()

        prefix = config.inference.prediction_prefix
        pyexr.write(str(output_dir / f"{prefix}_{sample_id}_amp.exr"), output[0])
        pyexr.write(str(output_dir / f"{prefix}_{sample_id}_phs.exr"), output[1])

    return output_dir
