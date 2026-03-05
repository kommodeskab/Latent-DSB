from src.lightning_modules import BaseLightningModule
from src import StepOutput, TensorDict, Batch
import torch
from typing import Optional


class BaseMetric:
    """
    This is a base class for metrics.
    Each metric should implement the following methods:
        - `add()`: called for each batch during validation. Should update the metric's internal state based on the model's outputs and the ground truth.
        - `compute()`: called at the end of each validation epoch. Should compute and return the final metric value based on the accumulated state from `add()`.
        - `reset()`: called at the end of each validation epoch after `compute()`. Should reset the metric's internal state for the next epoch.
        - `to()`: called at the start of training. Should move any internal tensors to the specified device.
        - `name()`: should return a string name for the metric, which will be used for logging.
    """

    def add(
        self,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        extras: Optional[TensorDict] = None,
    ): ...
    def compute(self) -> float: ...
    def reset(self) -> None: ...
    def to(self, device: torch.device) -> None: ...
    def name(self) -> str: ...
