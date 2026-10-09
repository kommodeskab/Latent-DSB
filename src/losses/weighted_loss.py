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

        counts: dict[str, int] = {}
        for loss_fn in losses:
            name = loss_fn.__class__.__name__
            counts[name] = counts.get(name, 0) + 1

        seen: dict[str, int] = {}
        self.function_names: list[str] = []
        for loss_fn in losses:
            name = loss_fn.__class__.__name__
            if counts[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                self.function_names.append(f"{name}{seen[name]}")
            else:
                self.function_names.append(name)
        self.loss_names: list[str] = self.function_names

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = LossOutput(loss=0.0)

        for loss_fn, weight, name in zip(self.losses, self.weights, self.function_names):
            loss_output: LossOutput = loss_fn(model_output, batch)
            loss["loss"] += weight * loss_output["loss"]

            # the loss outputs might contain other keys than "loss",
            # we also want to log these, therefore, we add the loss name as prefix to the key
            for key, value in loss_output.items():
                loss[f"{name}_{key}"] = value

        loss["loss"] = loss["loss"] / sum(self.weights)

        return loss
