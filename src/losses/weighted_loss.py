from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput
import torch.nn as nn
from typing import Optional


class WeightedLoss(BaseLossFunction):
    def __init__(
        self,
        losses: list[BaseLossFunction],
        weights: Optional[list[float] | float] = None,
    ):
        super().__init__()
        self.losses: list[BaseLossFunction] = nn.ModuleList(losses)
        # if weights is a single float, convert it to a list of the same length as losses
        # if it is none, set it to a list of 1.0s
        weights = 1.0 if weights is None else weights
        self.weights = [weights] * len(losses) if isinstance(weights, float) else weights

        assert len(self.losses) == len(self.weights), "Losses and weights must have the same length"

        # make a list of loss function names
        # each loss function name will have a suffix indicating its index among losses with the same name (e.g. Name_1, Name_2)
        self.loss_names = []
        name_counts = {}
        for loss in self.losses:
            base_name = loss.__class__.__name__
            count = name_counts.get(base_name, 0) + 1
            name_counts[base_name] = count
            self.loss_names.append(f"{base_name}_{count}")

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = {"loss": 0.0}

        for loss_index, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            loss_output = loss_fn(model_output, batch)
            loss["loss"] += weight * loss_output["loss"]

            # the loss outputs might contain other keys than "loss",
            # we also want to log these, therefore, we add the loss name as prefix to the key
            for key, value in loss_output.items():
                loss[f"{self.loss_names[loss_index]}_{key}"] = value

        return LossOutput(**loss)
