import torch.nn as nn
import torch
from omegaconf.listconfig import ListConfig

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_hat, x = batch["out"], batch["target"]
        loss = self.mse_loss.forward(x_hat, x)
        return {"loss": loss}

class SmooothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta)
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_hat, x = batch["out"], batch["target"]
        loss = self.smooth_l1_loss.forward(x_hat, x)
        return {"loss": loss}

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_hat, x = batch["out"], batch["target"]
        loss = self.l1_loss.forward(x_hat, x)
        return {"loss": loss}
    
class MultiChannelMSELoss(nn.Module):
    def __init__(
        self,
        weights: list[float] | float = 1.0,
        ):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.weights = weights
        
    def forward(
        self, 
        batch: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
        x_hat, x = batch["out"], batch["target"]
        n_channels = x.shape[1]
        loss_dict = {}
        loss = 0
        for c in range(n_channels):
            w = self.weights[c] if isinstance(self.weights, ListConfig) else self.weights
            loss_c = w * self.mse_loss.forward(x_hat[:, c], x[:, c])
            loss_dict[f"loss_channel_{c}"] = loss_c
            loss += loss_c
        loss /= n_channels
        loss_dict["loss"] = loss
        return loss_dict