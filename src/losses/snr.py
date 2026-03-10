from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput
import torch

class SNRLoss(BaseLossFunction):
    def __init__(self):
        super().__init__()
        
    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        output = model_output['output']
        target = batch['target']
        
        # Calculate the signal power and noise power
        signal_power = (target ** 2).mean()
        noise_power = ((target - output) ** 2).mean()
        # Calculate SNR in decibels
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-6))  # Adding a small value to avoid division by zero
        # We want to maximize SNR, so we return the negative SNR as the loss
        loss = -snr
        return LossOutput(loss=loss)