"""Physics-informed CGH model skeleton for KOREATECH-CGH dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CGHConfig:
    depth_range_mm: float = 80.0
    num_buckets: int = 8
    asm_wavelength_nm: float = 532.0
    pixel_pitch_um: float = 6.4
    use_swin: bool = True


def depth_frequency_encoding(
    depth_map: torch.Tensor,
    config: CGHConfig,
) -> torch.Tensor:
    """Encode propagation distance into spatial-frequency features.

    Args:
        depth_map: (B, 1, H, W) depth in mm.
        config: CGH configuration.

    Returns:
        (B, 2, H, W) frequency-domain encoding (cos/sin) for z.
    """
    device = depth_map.device
    b, _, h, w = depth_map.shape
    fy = torch.fft.fftfreq(h, d=config.pixel_pitch_um * 1e-6, device=device)
    fx = torch.fft.fftfreq(w, d=config.pixel_pitch_um * 1e-6, device=device)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    spatial_freq = torch.sqrt(fx**2 + fy**2)
    spatial_freq = spatial_freq.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)

    depth_m = depth_map * 1e-3
    phase = 2.0 * torch.pi * spatial_freq * depth_m
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)


def bucketize_depth(
    depth_map: torch.Tensor,
    config: CGHConfig,
) -> torch.Tensor:
    """Split RGB into depth buckets (feature buckets).

    Args:
        depth_map: (B, 1, H, W) depth in mm.
        config: CGH configuration.

    Returns:
        (B, num_buckets, H, W) bucket indicators.
    """
    device = depth_map.device
    edges = torch.linspace(
        0.0,
        config.depth_range_mm,
        config.num_buckets + 1,
        device=device,
    )
    indices = torch.bucketize(depth_map, edges, right=False) - 1
    indices = indices.clamp(0, config.num_buckets - 1)
    b, _, h, w = depth_map.shape
    one_hot = F.one_hot(indices.squeeze(1), num_classes=config.num_buckets)
    return one_hot.permute(0, 3, 1, 2).float()


def calculate_bidirectional_mask(
    depth_map: torch.Tensor,
    k: int = 2,
    *,
    depth_range_mm: float = 80.0,
    num_layers: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute bidirectional layer masks (Eq. 5 & 6 style).

    The mask aggregates forward and backward layers around each pixel's depth
    index to suppress contour aliasing and shadow artifacts.

    Args:
        depth_map: (B, 1, H, W) depth in mm.
        k: number of neighboring layers in each direction.
        depth_range_mm: maximum depth in mm.
        num_layers: total discrete layers.

    Returns:
        forward_mask: (B, num_layers, H, W)
        backward_mask: (B, num_layers, H, W)
        combined_mask: (B, num_layers, H, W)
    """
    device = depth_map.device
    edges = torch.linspace(0.0, depth_range_mm, num_layers + 1, device=device)
    layer_idx = torch.bucketize(depth_map, edges, right=False) - 1
    layer_idx = layer_idx.clamp(0, num_layers - 1)

    b, _, h, w = depth_map.shape
    forward_mask = torch.zeros(b, num_layers, h, w, device=device)
    backward_mask = torch.zeros_like(forward_mask)

    for offset in range(k + 1):
        f_idx = (layer_idx + offset).clamp(0, num_layers - 1)
        b_idx = (layer_idx - offset).clamp(0, num_layers - 1)
        forward_mask.scatter_(1, f_idx, 1.0)
        backward_mask.scatter_(1, b_idx, 1.0)

    combined_mask = torch.clamp(forward_mask + backward_mask, 0.0, 1.0)
    return forward_mask, backward_mask, combined_mask


class FIPLoss(nn.Module):
    """Physics-driven loss with focal image projection (FIP) aggregation."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, focal_stack: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Aggregate in-focus regions from the focal stack.

        Args:
            focal_stack: (B, L, H, W) focal reconstructions.
            target: (B, 1, H, W) target intensity.
        """
        sharpness = torch.abs(focal_stack)
        weights = sharpness / (sharpness.sum(dim=1, keepdim=True) + 1e-8)
        fip = (weights * focal_stack).sum(dim=1, keepdim=True)
        return self.weight * F.l1_loss(fip, target)


class PhysicsInformedCGH(nn.Module):
    """Hybrid Swin-Transformer + ASM model skeleton."""

    def __init__(self, config: CGHConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + config.num_buckets + 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, rgb: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Forward pass for hologram prediction."""
        depth_buckets = bucketize_depth(depth_map, self.config)
        freq_encoding = depth_frequency_encoding(depth_map, self.config)
        features = torch.cat([rgb, depth_buckets, freq_encoding], dim=1)
        encoded = self.encoder(features)
        hologram = self.decoder(encoded)
        return hologram
