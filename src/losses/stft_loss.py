import torch.nn as nn
from torch import Tensor
from typing import Dict
import torch.nn.functional as F
from typing import Optional
import torch

Data = Dict[str, Tensor]

class STFTLoss(nn.Module):
    def __init__(
        self, 
        weights : Optional[Dict[str, float]] = None
        ):
        super().__init__()
        self.weights = {'sc': 1.0, 'logmag': 1.0, 'complex': 0.1}
        if weights is not None:
            self.weights.update(weights)
        
    def forward(self, batch: Data) -> Tensor:
        x_hat, x = batch["out"], batch["target"]
        
        xr, xi = x[:,0], x[:,1]
        yr, yi = x_hat[:,0], x_hat[:,1] 
        
        _eps = 1e-8
        
        # magnitude
        mag_x = torch.sqrt(xr**2 + xi**2 + _eps)
        mag_xhat = torch.sqrt(yr**2 + yi**2 + _eps)
        
        log_mag_x = (mag_x + _eps).log()
        log_mag_xhat = (mag_xhat + _eps).log()
        
        # Spectral convergence
        num = torch.norm(mag_x - mag_xhat, p='fro', dim=(1,2))
        den = torch.norm(mag_x, p='fro', dim=(1,2)).clamp(min=_eps)
        loss_sc = (num / den).mean()
        
        # Log-magnitude
        loss_logmag = F.l1_loss(log_mag_x, log_mag_xhat)
        
        # phase aware loss
        loss_complex = F.l1_loss(x, x_hat)
        
        loss = (
            self.weights['sc'] * loss_sc +
            self.weights['logmag'] * loss_logmag +
            self.weights['complex'] * loss_complex
        )
        
        return loss