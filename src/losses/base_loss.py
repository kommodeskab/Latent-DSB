import torch.nn as nn
import torch

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Forward method not implemented in BaseLoss")