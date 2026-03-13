from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .encoders import KOREATECHCGHEncoder


class KOREATECHHolographyDataset(Dataset):
    def __init__(self, split_root: str | Path, encoder: KOREATECHCGHEncoder):
        self.split_root = Path(split_root)
        self.encoder = encoder
        self.img_dir = self.split_root / "img"
        self.depth_dir = self.split_root / "depth"
        self.amp_dir = self.split_root / "amp"
        self.phs_dir = self.split_root / "phs"
        self.indices = sorted(path.stem for path in self.img_dir.glob("*.exr"))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.indices[idx]
        x = self.encoder.encode(
            self.img_dir / f"{sample_id}.exr",
            self.depth_dir / f"{sample_id}.exr",
        )

        amp_raw = self.encoder.load_exr(self.amp_dir / f"{sample_id}.exr")
        phs_raw = self.encoder.load_exr(self.phs_dir / f"{sample_id}.exr")

        amp = torch.from_numpy(amp_raw)
        phs = torch.from_numpy(phs_raw)

        if amp.ndim == 3:
            amp = amp[:, :, 0]
        if phs.ndim == 3:
            phs = phs[:, :, 0]

        y = torch.cat([amp.unsqueeze(0), phs.unsqueeze(0)], dim=0).float()
        return x, y
