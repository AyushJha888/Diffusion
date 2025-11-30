import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal timestep embedding used in diffusion models.
    timesteps: (B,) int64 tensor
    returns: (B, dim) float tensor
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        # inject time information
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.dropout(self.act(h))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.res = ResidualBlock1D(in_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res(x, t_emb)
        skip = x
        x = F.interpolate(x, scale_factor=0.5, mode="linear", align_corners=False, recompute_scale_factor=False)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.res = ResidualBlock1D(in_channels + skip_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class PositionalUNet1D(nn.Module):
    """
    A light-weight 1D U-Net over time with timestep conditioning ("punet" from the assignment).
    Input/output shape: (B, C, T) where C = 2 * #keypoints.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 4),
        time_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.input_proj = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        # Down path
        down_channels: List[int] = [base_channels * m for m in channel_mults]
        downs: List[DownBlock] = []
        in_ch = base_channels
        for ch in down_channels:
            downs.append(DownBlock(in_ch, ch, time_dim))
            in_ch = ch
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.mid1 = ResidualBlock1D(in_ch, in_ch, time_dim, dropout=dropout)
        self.mid2 = ResidualBlock1D(in_ch, in_ch, time_dim, dropout=dropout)

        # Up path
        ups: List[UpBlock] = []
        for ch in reversed(down_channels):
            ups.append(UpBlock(in_ch, ch, ch, time_dim))
            in_ch = ch
        self.ups = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_proj = nn.Conv1d(in_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        timesteps: (B,) int tensor
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"Input must have shape (B, C, T); got {x.shape}")
        x = x.contiguous()
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.input_proj(x)
        skips: List[torch.Tensor] = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb)

        x = self.out_proj(self.act(self.out_norm(x)))
        return x


__all__ = ["PositionalUNet1D", "sinusoidal_embedding"]
