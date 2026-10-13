from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention import RotarySelfAttention
from .model import UNet1D, timestep_embedding


class RotaryTransformerBlock(nn.Module):


    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = RotarySelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MultiscaleAttentionUNet1D(UNet1D):


    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256, 256, 256, 256),
        time_dim: int = 256,
        num_res_blocks: int = 2,
        attention_heads: int = 4,
        gradient_checkpointing: bool = False,
        attention_levels: Sequence[int] = (3, 4),
        attention_depth: int = 2,
        bottleneck_depth: int = 4,
        mlp_ratio: float = 4.0,
    ):
        channels = tuple(channels)
        levels = tuple(attention_levels)
        if not isinstance(attention_heads, int) or isinstance(attention_heads, bool) or attention_heads < 1:
            raise ValueError("attention_heads must be a positive integer")
        if any(not isinstance(level, int) or isinstance(level, bool) for level in levels):
            raise ValueError("attention_levels must contain integer encoder indices")
        if len(set(levels)) != len(levels) or any(level < 0 or level >= len(channels) - 1 for level in levels):
            raise ValueError("attention_levels must be unique non-bottleneck encoder indices")
        for name, value in (("attention_depth", attention_depth), ("bottleneck_depth", bottleneck_depth)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(mlp_ratio, (int, float)) or isinstance(mlp_ratio, bool) or not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")

        super().__init__(channels, time_dim, num_res_blocks, attention_heads, gradient_checkpointing)
        for level in (*levels, len(channels) - 1):
            width = channels[level]
            if width % attention_heads or (width // attention_heads) % 2:
                raise ValueError("attention widths must be divisible by heads with an even head dimension for RoPE")
            if int(width * mlp_ratio) < 1:
                raise ValueError("mlp_ratio must produce at least one hidden feature")
        self.encoder_attention = nn.ModuleDict({
            str(level): nn.ModuleList(
                RotaryTransformerBlock(channels[level], attention_heads, mlp_ratio)
                for _ in range(attention_depth)
            )
            for level in sorted(levels)
        })
        self.attention = nn.ModuleList(
            RotaryTransformerBlock(channels[-1], attention_heads, mlp_ratio)
            for _ in range(bottleneck_depth)
        )
        self.model_config.update({
            "architecture": "unet_multiscale_attention",
            "attention_levels": sorted(levels),
            "attention_depth": attention_depth,
            "bottleneck_depth": bottleneck_depth,
            "mlp_ratio": float(mlp_ratio),
        })

    def _transformer_stack(self, blocks: nn.ModuleList, x: Tensor) -> Tensor:
        tokens = x.transpose(1, 2)
        for block in blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens.transpose(1, 2)

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
            if str(level) in self.encoder_attention:
                h = self._transformer_stack(self.encoder_attention[str(level)], h)
            if level < len(self.downsample):
                skips.append(h)
                h = self.downsample[level](h)
        h = self._transformer_stack(self.attention, h)
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
