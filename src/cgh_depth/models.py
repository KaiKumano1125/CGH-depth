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
    Lightweight parallel encoder for frequency (cos/sin) input channels.
    Mirrors the main encoder's 4 downsampling stages to produce features at
    the same bottleneck resolution (input_res / 16) for cross-attention.
    """

    def __init__(self, freq_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 4
        self.net = nn.Sequential(
            DoubleConv(freq_channels, mid),
            nn.MaxPool2d(2),
            DoubleConv(mid, mid * 2),
            nn.MaxPool2d(2),
            DoubleConv(mid * 2, mid * 4),
            nn.MaxPool2d(2),
            DoubleConv(mid * 4, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention at the UNet bottleneck.
      Q   : main encoder bottleneck features (x5), shape (B, C, H, W).
      K/V : frequency context from FrequencyContextEncoder, same shape.

    Both tensors are flattened to (B, H*W, C) token sequences before attention.
    Uses pre-norm, residual connections, and a position-wise feed-forward layer.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_seq = x.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        ctx_seq = context.flatten(2).permute(0, 2, 1)   # (B, H*W, C)

        # Cross-attention with residual
        attn_out, _ = self.attn(
            self.norm_q(x_seq),
            self.norm_kv(ctx_seq),
            self.norm_kv(ctx_seq),
        )
        x_seq = x_seq + attn_out

        # Feed-forward with residual
        x_seq = x_seq + self.ff(self.norm_ff(x_seq))

        return x_seq.permute(0, 2, 1).reshape(B, C, H, W)


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

        # Cross-attention at bottleneck — only built when frequency channels exist
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

        # Cross-attention: bottleneck queries frequency context
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

    # Compute which input channels are frequency (cos/sin) based on encoder channel ordering:
    # [RGB (0-2)] [depth (3)] [freq_cos (4)] [freq_sin (5)]
    freq_start = (3 if encoder_config.include_rgb else 0) + (1 if encoder_config.include_depth else 0)
    freq_count = int(encoder_config.include_freq_cos) + int(encoder_config.include_freq_sin)

    # Only wire up the separate frequency branch when explicitly requested.
    # Exp2 (concat) has freq channels but use_cross_attention=False → freq_slice=None.
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
