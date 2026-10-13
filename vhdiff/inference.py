from pathlib import Path

import torch
from torch import nn

from .model import ConditionalUNet1D
from .sampling import sample_batch


class DiffusionSchedule(nn.Module):
    def __init__(self, alpha_bar, config):
        super().__init__()
        self.timesteps = config["timesteps"]
        self.prediction_type = config["prediction_type"]
        self.data_scale = config["data_scale"]
        self.register_buffer("alpha_bar", alpha_bar.float().clone())

    def ddim_timesteps(self, steps):
        return torch.linspace(0, self.timesteps-1, steps).round().long().flip(0).tolist()


def load_model(checkpoint, device="cuda"):
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if payload.get("format") != "vhdiff_inference_v1":
        raise ValueError("Use the released checkpoints/vhdiff.pt inference weights")
    if payload["label_order"] != ["AS", "AR", "MS", "MR", "TR"]:
        raise ValueError("Unsupported disease attribute order")
    model = ConditionalUNet1D(payload["model_config"], payload["condition_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    diffusion = DiffusionSchedule(payload["alpha_bar"], payload["diffusion_config"]).to(device)
    return model, diffusion, dict(payload["default_sampler"])


def generate_batch(model, diffusion, requests, sampler, device):

    return sample_batch(model, diffusion, requests, sampler, device)
