import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch):
        x_hat, x = batch["out"], batch["target"]
        loss = self.mse_loss(x_hat, x)
        return loss