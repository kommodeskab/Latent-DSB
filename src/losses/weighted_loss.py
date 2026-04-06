from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput
import torch.nn as nn


class WeightedLoss(BaseLossFunction):
    def __init__(
        self,
        losses: list[BaseLossFunction],
        weights: list[float],
    ):
        super().__init__()
        assert len(losses) == len(weights), "Losses and weights must have the same length"
        self.losses: list[BaseLossFunction] = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = {"loss": 0.0}

        for loss_fn, weight in zip(self.losses, self.weights):
            loss_output = loss_fn.forward(model_output, batch)
            loss["loss"] += weight * loss_output["loss"]

            # the loss outputs might contain other keys than "loss",
            # we also want to log these, therefore, we add the loss name as prefix to the key
            for key, value in loss_output.items():
                loss[f"{loss_fn.__class__.__name__}_{key}"] = value

        return LossOutput(**loss)
