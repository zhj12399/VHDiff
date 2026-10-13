from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RotarySelfAttention(nn.Module):


    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not isinstance(num_heads, int) or num_heads < 1 or dim % num_heads:
            raise ValueError("dim must be divisible by a positive num_heads")
        head_dim = dim // num_heads
        if head_dim % 2:
            raise ValueError("RoPE requires an even dimension per attention head")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def _rotate(self, tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        pairs = tensor.float().reshape(*tensor.shape[:-1], self.head_dim // 2, 2)
        left, right = pairs.unbind(dim=-1)
        rotated = torch.stack((left * cos - right * sin, left * sin + right * cos), dim=-1)
        return rotated.flatten(-2).to(tensor.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim or x.shape[1] < 1:
            raise ValueError(f"x must have shape [batch, length >= 1, {self.dim}]")
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)


        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
            / self.head_dim
        )
        phases = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * frequencies
        cos, sin = phases.cos()[None, None], phases.sin()[None, None]
        q = self._rotate(q, cos, sin)
        k = self._rotate(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, self.dim)
        return self.proj(attended)
