from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .base.model import ResBlock1D, UNet1D, timestep_embedding
from .base.multiscale_unet import MultiscaleAttentionUNet1D


NUM_ATTRIBUTES = 5
LABEL_ORDER = ("AS", "AR", "MS", "MR", "TR")


def normalize_conditions(
    values: Tensor | None, known: Tensor | None, batch: int, device: torch.device
) -> tuple[Tensor, Tensor]:

    if values is None and known is None:
        empty = torch.zeros((batch, NUM_ATTRIBUTES), device=device, dtype=torch.long)
        return empty, empty.clone()
    if values is None or known is None:
        raise ValueError("values and known must both be supplied, or both omitted")
    values = torch.as_tensor(values, device=device)
    known = torch.as_tensor(known, device=device)
    for name, tensor in (("values", values), ("known", known)):
        if tensor.ndim not in (1, 2) or tensor.shape[-1] != NUM_ATTRIBUTES:
            raise ValueError(f"{name} must have shape [5], [1,5] or [batch,5]")
        if tensor.ndim == 2 and tensor.shape[0] not in (1, batch):
            raise ValueError(f"{name} batch does not match waveforms")
    values = values.reshape(-1, NUM_ATTRIBUTES).expand(batch, -1)
    known = known.reshape(-1, NUM_ATTRIBUTES).expand(batch, -1)
    if not bool(((known == 0) | (known == 1)).all()):
        raise ValueError("known must contain only zero or one")
    if not bool((((values == 0) | (values == 1)) | (known == 0)).all()):
        raise ValueError("known label values must contain only zero or one")
    values = torch.where(known.bool(), values, torch.zeros_like(values)).long()
    return values, known.long()


class ConditionEncoder(nn.Module):


    def __init__(self, embedding_dim: int, time_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(3, embedding_dim) for _ in LABEL_ORDER)
        self.mlp = nn.Sequential(
            nn.Linear(NUM_ATTRIBUTES * embedding_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    @staticmethod
    def state_indices(values: Tensor, known: Tensor) -> Tensor:
        return torch.where(known.bool(), values.long(), torch.full_like(values.long(), 2))

    def forward(self, values: Tensor, known: Tensor) -> Tensor:
        states = self.state_indices(values, known)
        encoded = torch.cat([
            embedding(states[:, attribute])
            for attribute, embedding in enumerate(self.embeddings)
        ], dim=-1)
        return self.mlp(encoded)


class LocalResidual(nn.Module):

    def __init__(self, channels, rank, kernel_size):
        super().__init__()
        self.down = nn.Conv1d(channels, rank, 1)
        self.kernel = nn.Conv1d(rank, rank, kernel_size, padding=kernel_size // 2, groups=rank)
        self.up = nn.Conv1d(rank, channels, 1)

    def forward(self, x):
        return self.up(F.silu(self.kernel(F.silu(self.down(x)))))


class ResidualAdaptation(nn.Module):


    def __init__(self, channels: int, time_dim: int, adapter_dim: int):
        super().__init__()
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels * 2))
        self.down = nn.Conv1d(channels, adapter_dim, kernel_size=1)
        self.up = nn.Conv1d(adapter_dim, channels, kernel_size=1)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self, block: ResBlock1D, x: Tensor, time_embedding: Tensor,
        condition_embedding: Tensor,
    ) -> Tensor:

        h = block.conv1(F.silu(block.norm1(x)))
        scale, shift = block.time_affine(time_embedding).to(h.dtype).chunk(2, dim=1)
        delta_scale, delta_shift = self.film(condition_embedding).to(h.dtype).chunk(2, dim=1)
        scale, shift = scale + delta_scale, shift + delta_shift
        h = block.norm2(h) * (1 + scale[:, :, None]) + shift[:, :, None]
        result = block.skip(x) + block.conv2(F.silu(h))
        result = result + self.up(F.silu(self.down(result)))
        extension = self._modules.get("extension")
        return result if extension is None else result + extension(result)


class ConditionalUNet1D(nn.Module):


    def __init__(self, model_config: dict, condition_config: dict):
        super().__init__()
        supported = {"embedding_dim", "adapter_dim", "temporal_rank", "temporal_kernel"}
        if set(condition_config) - supported:
            raise ValueError(f"unknown condition configuration: {sorted(set(condition_config) - supported)}")
        self.condition_config = {"embedding_dim": 32, "adapter_dim": 32, **condition_config}
        for name, value in self.condition_config.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        base_config = dict(model_config)
        self.architecture = base_config.pop("architecture", "unet")
        if self.architecture == "unet":
            self.base = UNet1D(**base_config)
        elif self.architecture == "unet_multiscale_attention":
            self.base = MultiscaleAttentionUNet1D(**base_config)
        else:
            raise ValueError(f"unsupported conditional backbone: {self.architecture!r}")
        self.base.requires_grad_(False)
        self.model_config = dict(self.base.model_config)
        self.gradient_checkpointing = self.base.gradient_checkpointing
        self.condition_encoder = ConditionEncoder(
            self.condition_config["embedding_dim"], self.base.time_dim
        )
        self.encoder_adapters = nn.ModuleList(nn.ModuleList(
            ResidualAdaptation(block.conv2.out_channels, self.base.time_dim,
                               self.condition_config["adapter_dim"])
            for block in blocks
        ) for blocks in self.base.encoder)
        self.decoder_adapters = nn.ModuleList(nn.ModuleList(
            ResidualAdaptation(block.conv2.out_channels, self.base.time_dim,
                               self.condition_config["adapter_dim"])
            for block in blocks
        ) for blocks in self.base.decoder)

        rank = self.condition_config.get("temporal_rank")
        kernel = self.condition_config.get("temporal_kernel")
        if rank is None or kernel is None or kernel % 2 == 0:
            raise ValueError("The released checkpoint requires temporal_rank and an odd temporal_kernel")
        for family in (self.encoder_adapters, self.decoder_adapters):
            for level in family:
                for adapter in level:
                    adapter.extension = LocalResidual(adapter.up.out_channels, rank, kernel)

    def _residual(
        self, block: ResBlock1D, adapter: ResidualAdaptation, x: Tensor,
        embedding: Tensor, condition: Tensor,
    ) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():

            return checkpoint(adapter, block, x, embedding, condition, use_reentrant=False)
        return adapter(block, x, embedding, condition)

    def forward(
        self, x: Tensor, t: Tensor, values: Tensor | None = None, known: Tensor | None = None
    ) -> Tensor:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] < 2:
            raise ValueError("x must have shape [batch,1,length], with length >= 2")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must contain one timestep per waveform")
        values, known = normalize_conditions(values, known, x.shape[0], x.device)
        condition = self.condition_encoder(values, known)
        embedding = timestep_embedding(t.to(x.device), self.base.time_dim)
        embedding = self.base.time_mlp(embedding.to(self.base.time_mlp[0].weight.dtype))
        h = self.base.input_conv(x)
        skips = []
        for level, (blocks, adapters) in enumerate(zip(self.base.encoder, self.encoder_adapters)):
            for block, adapter in zip(blocks, adapters):
                h = self._residual(block, adapter, h, embedding, condition)


            if self.architecture == "unet_multiscale_attention":
                if str(level) in self.base.encoder_attention:
                    h = self.base._transformer_stack(self.base.encoder_attention[str(level)], h)
            if level < len(self.base.downsample):
                skips.append(h)
                h = self.base.downsample[level](h)
        if self.architecture == "unet_multiscale_attention":
            h = self.base._transformer_stack(self.base.attention, h)
        elif self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            h = checkpoint(self.base.attention, h, use_reentrant=False)
        else:
            h = self.base.attention(h)
        for skip, upsample, blocks, adapters in zip(
            reversed(skips), self.base.upsample_conv, self.base.decoder, self.decoder_adapters
        ):
            h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = upsample(h)
            h = torch.cat((h, skip), dim=1)
            for block, adapter in zip(blocks, adapters):
                h = self._residual(block, adapter, h, embedding, condition)
        output = self.base.output_conv(F.silu(self.base.output_norm(h)))
        if output.shape != x.shape:
            raise RuntimeError("conditional U-Net unexpectedly changed waveform shape")
        return output
