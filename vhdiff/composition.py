from __future__ import annotations
import torch
from torch import Tensor, nn
from .model import NUM_ATTRIBUTES, normalize_conditions

class CompositePredictor(nn.Module):


    def __init__(self, model: nn.Module, values: Tensor, known: Tensor):
        super().__init__()
        self.model = model


        self.training = model.training
        self.register_buffer("values", torch.as_tensor(values).clone())
        self.register_buffer("known", torch.as_tensor(known).clone())

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        values, known = normalize_conditions(self.values, self.known, x.shape[0], x.device)
        null_mask = torch.zeros_like(known)
        null = self.model(x, t, torch.zeros_like(values), null_mask).float()
        count = known.sum(1)
        result = null.clone()
        single = null.clone()
        for attribute in range(NUM_ATTRIBUTES):
            active = known[:, attribute].bool()
            if not bool(active.any()):
                continue
            attribute_mask = torch.zeros_like(known)
            attribute_mask[:, attribute] = known[:, attribute]
            prediction = self.model(x, t, values * attribute_mask, attribute_mask).float()
            active = active[:, None, None]
            result = result + torch.where(active, prediction - null, torch.zeros_like(null))
            single = torch.where(active, prediction, single)

        return torch.where((count == 1)[:, None, None], single, result)
