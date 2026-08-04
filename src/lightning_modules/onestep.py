import torch
from torch import Tensor
import torch.nn as nn
from src.lightning_modules import BaseLightningModule
from src.losses import BaseLossFunction
from src import OptimizerType, LRSchedulerType, ModelOutput, StepOutput, UnpairedAudioBatch


class OneStepModel(BaseLightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: BaseLossFunction,
        sigma: tuple[float, float] = (0.0, 0.0),
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.model = model
        self.sigma_min, self.sigma_max = sigma
        assert self.sigma_min <= self.sigma_max, "sigma_min must be less than or equal to sigma_max"
        self.network = model  # Alias for compatibility with network/model
        self.loss_fn = loss_fn

    def forward(self, batch: UnpairedAudioBatch) -> ModelOutput:
        output = self.model(batch["x1"])
        return ModelOutput(output=output)

    def common_step(self, batch: UnpairedAudioBatch, batch_idx: int) -> StepOutput:
        batch["target"] = batch["x0"]  # add a target key to the batch for the loss function

        sigma = (
            torch.rand(batch["x1"].shape[0], device=batch["x1"].device, dtype=batch["x1"].dtype)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        sigma = sigma.view(-1, *([1] * (batch["x1"].ndim - 1)))
        batch["x1"] = batch["x1"] + sigma * torch.randn_like(batch["x1"])  # inject noise into the input batch

        output = self(batch)
        loss = self.loss_fn(output, batch)
        return StepOutput(
            loss=loss["loss"],
            loss_output=loss,
            model_output=output,
            module=self,
        )

    @torch.no_grad()
    def sample(self, x_start: Tensor, **kwargs) -> Tensor:
        return self.model(x_start)
