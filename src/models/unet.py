import math

import torch
from einops import rearrange
from torch import nn


class FourierEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = rearrange(t, "b -> b 1")
        freqs = t * self.weights * 2 * math.pi  # b hd
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # b d


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act = Swish()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        # time_emb not used in this simple UNet, but kept for compatibility
        x_skip = self.skip(x) if self.in_channels != self.out_channels else x
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return x_skip + h


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(2)
        # Time embedding projection (optional)
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(time_emb_dim, out_channels),
                Swish(),
                nn.Linear(out_channels, out_channels),
            )
            if time_emb_dim > 0
            else None
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        if self.time_mlp is not None:
            scale_shift = self.time_mlp(t_emb)[..., None, None]
            x = x * (1 + scale_shift)  # simple modulation
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res = ResidualBlock(in_channels + out_channels, out_channels, dropout)
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(time_emb_dim, out_channels),
                Swish(),
                nn.Linear(out_channels, out_channels),
            )
            if time_emb_dim > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        if self.time_mlp is not None:
            scale_shift = self.time_mlp(t_emb)[..., None, None]
            x = x * (1 + scale_shift)
        return x


class UNet(nn.Module):
    """Simple attention-free UNet for 28x28 single-channel images (MNIST).

    Resolution path: 28 -> (down1) 14 -> (down2) 7 -> bottleneck -> (up1) 14 -> (up2) 28.
    Channel path:    32 ->    32   ->     64      ->     64      ->    32   ->    32   -> 1
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        time_emb_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_encoder = FourierEncoder(time_emb_dim)
        self.inc = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        # Down
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim, dropout)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim, dropout)
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, dropout)
        # Up
        self.up1 = UpBlock(base_channels * 2, base_channels, time_emb_dim, dropout)
        self.up2 = UpBlock(base_channels, base_channels, time_emb_dim, dropout)
        self.out = nn.Sequential(
            nn.GroupNorm(1, base_channels),
            Swish(),
            nn.Conv2d(base_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_encoder(t)
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x3 = self.bottleneck(x3)
        x = self.up1(x3, x2, t_emb)
        x = self.up2(x, x1, t_emb)
        return self.out(x)
