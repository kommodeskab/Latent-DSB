from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput
import torch.nn as nn


class SmoothL1Loss(BaseLossFunction):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta)

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = self.smooth_l1_loss.forward(model_output["output"], batch["target"])
        return LossOutput(loss=loss)


class L1Loss(BaseLossFunction):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = self.l1_loss.forward(model_output["output"], batch["target"])
        return LossOutput(loss=loss)
