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


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
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
    return SimpleUNet(
        in_channels=encoder_config.in_channels,
        out_channels=model_config.out_channels,
        base_channels=model_config.base_channels,
    )
