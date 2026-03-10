from src.lightning_modules import BaseLightningModule
from src.losses import BaseLossFunction
import torch.nn as nn
from src import OptimizerType, LRSchedulerType, Batch, ModelOutput, StepOutput


class DummyModule(BaseLightningModule):
    """A dummy LightningModule for testing purposes."""

    def __init__(
        self,
        network: nn.Module,
        loss_fn: BaseLossFunction,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None,
    ):
        # optimizer and lr_scheduler can be None, for example during testing
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.network = network
        self.loss_fn = loss_fn

    def forward(self, batch: Batch) -> ModelOutput:
        output = self.network(batch["input"])
        return ModelOutput(output=output)

    def common_step(self, batch: Batch, batch_idx: int) -> StepOutput:
        output = self.forward(batch)
        loss = self.loss_fn(output, batch)
        return StepOutput(
            loss=loss["loss"],
            loss_output=loss,
            model_output=output,
            module=self,
        )
