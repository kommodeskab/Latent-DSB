import torch.nn as nn
from torch import Tensor
from typing import Dict
import torch.nn.functional as F
from typing import Optional, Literal
import torch

Data = Dict[str, Tensor]

class STFTLoss(nn.Module):
    def __init__(
        self, 
        weights : Optional[Dict[str, float]] = None,
        polar_encoding : bool = False,
        complex_mode : Literal['l1', 'cos']  = 'l1'
        ):
        super().__init__()
        self.weights = {'sc': 1.0, 'logmag': 1.0, 'complex': 0.1}
        if weights is not None:
            self.weights.update(weights)
        self.polar_encoding = polar_encoding
        self.complex_mode = complex_mode
        assert complex_mode in ['l1', 'cos'], "complex_mode must be 'l1' or 'cos'"
            
    def stft_features(self, x : Tensor) -> Data:
        real, imag = x[:,0], x[:,1]
        mag = torch.sqrt(real**2 + imag**2 + 1e-5)
        log_mag = torch.log(mag + 1e-5)
        phase = torch.atan2(imag, real)
        return {
            'mag': mag,
            'log_mag': log_mag,
            'phase': phase,
            'stft': x,
        }
            
    def polar_features(self, x : Tensor) -> Data:
        # a polar encoding is a 3-channel representation of the STFT
        # first channel is the log-magnitude, second and third channels are the cosine and sine of the phase
        log_mag = x[:, 0].clamp(max=10) # avoid too large values for numerical stability since we exponentiate later
        mag = log_mag.exp()
        cos_phase = x[:, 1]
        sin_phase = x[:, 2]
        phase = torch.atan2(sin_phase, cos_phase)
        stft = torch.stack([mag * cos_phase, mag * sin_phase], dim=1)
        return {
            'mag': mag,
            'log_mag': log_mag,
            'phase': phase,
            'stft': stft,
        }
        
    def forward(self, batch: Data) -> Tensor:
        x_hat, x = batch["out"], batch["target"]
        
        if self.polar_encoding:
            features_x = self.polar_features(x)
            features_xhat = self.polar_features(x_hat)
        else:
            features_x = self.stft_features(x)
            features_xhat = self.stft_features(x_hat)
        
        # spectral convergence
        num = torch.norm(features_x['mag'] - features_xhat['mag'], p='fro', dim=(1,2))
        den = torch.norm(features_x['mag'], p='fro', dim=(1,2)).clamp(min=1e-5)
        loss_sc = (num / den).mean()
        
        # log-magnitude
        loss_logmag = F.l1_loss(features_x['log_mag'], features_xhat['log_mag'])
        
        # phase aware loss
        if self.complex_mode == 'cos':
            phi_x = features_x['phase']
            phi_y = features_xhat['phase']
            loss_phase = (1 - torch.cos(phi_x - phi_y)).mean()
        elif self.complex_mode == 'l1':
            loss_phase = F.l1_loss(features_x['stft'], features_xhat['stft'])
                
        loss = (
            self.weights['sc'] * loss_sc +
            self.weights['logmag'] * loss_logmag +
            self.weights['complex'] * loss_phase
        )
        
        return {
            "sc_loss": loss_sc,
            "logmag_loss": loss_logmag,
            "complex_loss": loss_phase,
            "loss": loss
        }