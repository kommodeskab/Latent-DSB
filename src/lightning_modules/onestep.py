import torch
from torch import Tensor
import torch.nn as nn
from src.lightning_modules import BaseLightningModule
from src.losses import BaseLossFunction
from src import OptimizerType, LRSchedulerType, Batch, ModelOutput, StepOutput


class OneStepModel(BaseLightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: BaseLossFunction,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.model = model
        self.network = model  # Alias for compatibility with network/model
        self.loss_fn = loss_fn

    def forward(self, batch: Batch) -> ModelOutput:
        output = self.model(batch["x1"])
        return ModelOutput(output=output)

    def common_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        batch["target"] = batch["x0"]  # add a target key to the batch for the loss function
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
