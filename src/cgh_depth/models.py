from __future__ import annotations

import torch
import torch.nn as nn

from .config import EncoderConfig, ModelConfig


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class FrequencyContextEncoder(nn.Module):
    """
    Lightweight frequency context encoder (Option C).
    Replaces the heavy 4-stage parallel UNet encoder with a single
    AdaptiveAvgPool + Conv1x1 projection.
    cos/sin (B, 2, 512, 512) → pool to (B, 2, 32, 32) → conv → (B, C, 32, 32)
    """

    def __init__(self, freq_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(freq_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Lightweight cross-attention at the UNet bottleneck (Option A + B).

    Option A — pool_factor=4: attend on 8×8=64 tokens instead of 32×32=1024.
               Reduces attention cost by (1024/64)² = 256×.
    Option B — inner_dim: project channels down for Q/K/V (e.g. 1024→256),
               then project back. Reduces attention op size 4×.

    After attention, upsampled result is added as residual to full-res x5.
    """

    def __init__(self, channels: int, num_heads: int = 4, pool_factor: int = 4, inner_dim: int = 256):
        super().__init__()
        self.pool_factor = pool_factor
        self.pool = nn.AvgPool2d(pool_factor)
        self.upsample = nn.Upsample(scale_factor=pool_factor, mode="bilinear", align_corners=False)

        # Project down to inner_dim for Q/K/V (Option B)
        self.proj_q = nn.Linear(channels, inner_dim)
        self.proj_kv = nn.Linear(channels, inner_dim)
        self.proj_out = nn.Linear(inner_dim, channels)

        self.norm_q = nn.LayerNorm(inner_dim)
        self.norm_kv = nn.LayerNorm(inner_dim)
        self.attn = nn.MultiheadAttention(inner_dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(inner_dim)
        self.ff = nn.Sequential(
            nn.Linear(inner_dim, inner_dim * 4),
            nn.GELU(),
            nn.Linear(inner_dim * 4, inner_dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Option A: pool 32×32 → 8×8 (64 tokens)
        x_pooled = self.pool(x)
        ctx_pooled = self.pool(context)

        x_seq = x_pooled.flatten(2).permute(0, 2, 1)       # (B, 64, C)
        ctx_seq = ctx_pooled.flatten(2).permute(0, 2, 1)    # (B, 64, C)

        # Option B: project to inner_dim
        q = self.proj_q(x_seq)      # (B, 64, inner_dim)
        kv = self.proj_kv(ctx_seq)  # (B, 64, inner_dim)

        # Cross-attention with residual
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + attn_out

        # Feed-forward with residual
        q = q + self.ff(self.norm_ff(q))

        # Project back to full channels
        x_refined = self.proj_out(q)  # (B, 64, C)

        # Reshape and upsample back to original resolution
        h_small = H // self.pool_factor
        w_small = W // self.pool_factor
        x_refined = x_refined.permute(0, 2, 1).reshape(B, C, h_small, w_small)
        x_refined = self.upsample(x_refined)   # 8×8 → 32×32

        return x + x_refined


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        freq_channel_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c4, c5))

        # Lightweight cross-attention at bottleneck
        self.freq_channel_slice = freq_channel_slice
        if freq_channel_slice is not None:
            freq_ch = freq_channel_slice[1] - freq_channel_slice[0]
            self.freq_encoder = FrequencyContextEncoder(freq_ch, c5)
            self.cross_attn = CrossAttentionBlock(c5)
        else:
            self.freq_encoder = None
            self.cross_attn = None

        self.up1 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(c4 + c4, c4)

        self.up2 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(c3 + c3, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(c2 + c2, c2)

        self.up4 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(c1 + c1, c1)

        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Lightweight cross-attention: bottleneck queries frequency context
        if self.freq_encoder is not None and self.freq_channel_slice is not None:
            freq_x = x[:, self.freq_channel_slice[0]:self.freq_channel_slice[1]]
            freq_context = self.freq_encoder(freq_x)
            x5 = self.cross_attn(x5, freq_context)

        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)

        return self.outc(u4)


def build_model(model_config: ModelConfig, encoder_config: EncoderConfig) -> nn.Module:
    if model_config.name != "simple_unet":
        raise ValueError(f"Unsupported model: {model_config.name}")

    freq_start = (3 if encoder_config.include_rgb else 0) + (1 if encoder_config.include_depth else 0)
    freq_count = int(encoder_config.include_freq_cos) + int(encoder_config.include_freq_sin)

    freq_slice = (
        (freq_start, freq_start + freq_count)
        if freq_count > 0 and model_config.use_cross_attention
        else None
    )

    return SimpleUNet(
        in_channels=encoder_config.in_channels,
        out_channels=model_config.out_channels,
        base_channels=model_config.base_channels,
        freq_channel_slice=freq_slice,
    )
