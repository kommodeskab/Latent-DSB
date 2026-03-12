from src.lightning_modules import BaseLightningModule
from src import StepOutput, TensorDict, Batch
import torch


class ExtraMetricOutput:
    """
    This is a base class for calculating and adding auxilary information to the metrics.
    For example, we might want to calculate some very specific samples which can be used to calculate some metrics.
    The __call__ method should return a dictionary of tensors. Theis dictionary will then be accessible by the metrics after
    each validation/testing step.
    """

    def __call__(
        self, pl_module: BaseLightningModule, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> TensorDict: ...
    def to(self, device: torch.device) -> None: ...
