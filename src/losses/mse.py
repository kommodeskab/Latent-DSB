from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput
import torch


class MSELoss(BaseLossFunction):
    """
    Simple example of how to implement Mean Squared Error Loss
    """

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = self.mse.forward(model_output["output"], batch["target"])
        return LossOutput(loss=loss)
