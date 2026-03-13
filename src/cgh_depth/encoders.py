from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import EncoderConfig


def _load_pyexr():
    try:
        import pyexr  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyexr is required for EXR loading. Use the project virtual environment "
            "(`.venv\\Scripts\\python.exe`) or install `pyexr` in the active interpreter."
        ) from exc
    return pyexr


class KOREATECHCGHEncoder:
    def __init__(self, config: EncoderConfig):
        self.config = config
        frequency = torch.fft.fftfreq(config.res, d=config.pitch)
        self.fy, self.fx = torch.meshgrid(frequency, frequency, indexing="ij")
        term = (1.0 / config.wavelength) ** 2 - self.fx**2 - self.fy**2
        self.phase_kernel = 2.0 * np.pi * torch.sqrt(torch.clamp(term, min=0))

    def load_exr(self, path: str | Path) -> np.ndarray:
        pyexr = _load_pyexr()
        return pyexr.open(str(path)).get().astype(np.float32)

    def encode(self, img_path: str | Path, depth_path: str | Path) -> torch.Tensor:
        rgb_raw = self.load_exr(img_path)
        depth_raw = self.load_exr(depth_path)

        features: list[torch.Tensor] = []

        if self.config.include_rgb:
            if rgb_raw.ndim == 3 and rgb_raw.shape[2] > 3:
                rgb_raw = rgb_raw[:, :, :3]
            rgb = (
                torch.from_numpy(rgb_raw).permute(2, 0, 1)
                if rgb_raw.ndim == 3
                else torch.from_numpy(rgb_raw).unsqueeze(0)
            )
            features.append(rgb.float())

        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth = torch.from_numpy(depth_raw).squeeze().float()
        z_map = depth * self.config.depth_range_m
        encoding_arg = self.phase_kernel * z_map

        if self.config.include_depth:
            features.append(depth.unsqueeze(0))
        if self.config.include_freq_cos:
            features.append(torch.cos(encoding_arg).unsqueeze(0))
        if self.config.include_freq_sin:
            features.append(torch.sin(encoding_arg).unsqueeze(0))

        return torch.cat(features, dim=0)
