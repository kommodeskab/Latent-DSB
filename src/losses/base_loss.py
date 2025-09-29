import torch.nn as nn

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch):
        raise NotImplementedError("Forward method not implemented in BaseLoss")