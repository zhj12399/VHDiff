from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _group_count(channels: int) -> int:

    return next(n for n in range(min(32, channels), 0, -1) if channels % n == 0)


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:

    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    phases = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((phases.cos(), phases.sin()), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.time_affine = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )


        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_affine(time_embedding).to(h.dtype).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None]) + shift[:, :, None]
        return self.skip(x) + self.conv2(F.silu(h))


class Attention1D(nn.Module):


    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, length = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            batch, 3, self.num_heads, channels // self.num_heads, length
        )
        q, k, v = qkv.unbind(dim=1)

        attended = F.scaled_dot_product_attention(
            q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1),
            dropout_p=0.0,
        )
        attended = attended.transpose(-2, -1).reshape(batch, channels, length)
        return x + self.output(attended)


class UNet1D(nn.Module):


    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256, 256, 256, 256),
        time_dim: int = 256,
        num_res_blocks: int = 2,
        attention_heads: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        channels = tuple(channels)
        if len(channels) < 2 or any(not isinstance(c, int) or c < 1 for c in channels):
            raise ValueError("channels must contain at least two positive integers")
        if time_dim < 2 or num_res_blocks < 1:
            raise ValueError("time_dim must be >= 2 and num_res_blocks must be positive")
        if attention_heads < 1 or channels[-1] % attention_heads:
            raise ValueError("bottleneck channels must be divisible by attention_heads")
        self.time_dim = time_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.model_config = {
            "channels": list(channels),
            "time_dim": time_dim,
            "num_res_blocks": num_res_blocks,
            "attention_heads": attention_heads,
            "gradient_checkpointing": gradient_checkpointing,
        }
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=5, padding=2)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 4 * time_dim), nn.SiLU(), nn.Linear(4 * time_dim, time_dim)
        )
        self.encoder = nn.ModuleList(
            nn.ModuleList(ResBlock1D(c, c, time_dim) for _ in range(num_res_blocks))
            for c in channels
        )
        self.downsample = nn.ModuleList(
            nn.Conv1d(left, right, kernel_size=5, stride=2, padding=2)
            for left, right in zip(channels[:-1], channels[1:])
        )
        self.attention = Attention1D(channels[-1], attention_heads)
        self.upsample_conv = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for level in range(len(channels) - 2, -1, -1):
            width = channels[level]
            self.upsample_conv.append(nn.Conv1d(channels[level + 1], width, 5, padding=2))
            self.decoder.append(nn.ModuleList([
                ResBlock1D(width * 2, width, time_dim),
                *(ResBlock1D(width, width, time_dim) for _ in range(num_res_blocks - 1)),
            ]))
        self.output_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.output_conv = nn.Conv1d(channels[0], 1, kernel_size=5, padding=2)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _residual(self, block: nn.Module, x: Tensor, embedding: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, embedding, use_reentrant=False)
        return block(x, embedding)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] < 2:
            raise ValueError("x must have shape [batch, 1, length], with length >= 2")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must contain one timestep per waveform")
        original_length = x.shape[-1]
        embedding = timestep_embedding(t.to(x.device), self.time_dim)
        embedding = self.time_mlp(embedding.to(self.time_mlp[0].weight.dtype))
        h = self.input_conv(x)
        skips = []
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                h = self._residual(block, h, embedding)
            if level < len(self.downsample):
                skips.append(h)
                h = self.downsample[level](h)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            h = checkpoint(self.attention, h, use_reentrant=False)
        else:
            h = self.attention(h)
        for skip, upsample, blocks in zip(reversed(skips), self.upsample_conv, self.decoder):
            h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = upsample(h)
            h = torch.cat((h, skip), dim=1)
            for block in blocks:
                h = self._residual(block, h, embedding)
        output = self.output_conv(F.silu(self.output_norm(h)))
        if output.shape != (x.shape[0], 1, original_length):
            raise RuntimeError("U-Net unexpectedly changed waveform shape")
        return output
