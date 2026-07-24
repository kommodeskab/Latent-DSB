from typing import Union, Tuple
import torch
import torch.nn as nn
from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput


class PowerLoss(BaseLossFunction):
    """
    Computes the L1 loss between the average power (mean squared magnitude)
    of the predicted and groundtruth waveforms.

    When `log_scale=True` (default), power is converted to decibels:
        P_dB = 10 * log10(P + eps)
    Matching power in the dB domain aligns with human logarithmic loudness perception.

    Args:
        dim: Dimension(s) along which to compute power (default: -1 for time dimension).
        log_scale: If True (default), computes L1 loss in decibels (dB).
        eps: Epsilon value for numerical stability in log calculation.
    """

    def __init__(
        self,
        dim: Union[int, Tuple[int, ...]] = -1,
        log_scale: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.log_scale = log_scale
        self.l1_loss = nn.L1Loss()

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        output = model_output["output"]
        target = batch["target"]

        # Compute power (mean squared energy) along specified dimension(s)
        pred_power = (output**2).mean(dim=self.dim)
        target_power = (target**2).mean(dim=self.dim)

        if self.log_scale:
            # Convert power to decibels (dB): 10 * log10(P + eps)
            pred_power = 10 * torch.log10(pred_power + 1e-7)
            target_power = 10 * torch.log10(target_power + 1e-7)

        loss = self.l1_loss(pred_power, target_power)
        return LossOutput(loss=loss)
