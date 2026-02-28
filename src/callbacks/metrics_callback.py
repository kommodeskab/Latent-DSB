from pytorch_lightning import Callback
from typing import Optional
from src import StepOutput, TensorDict, Batch
from src.lightning_modules import BaseLightningModule
import pytorch_lightning as pl
import torch


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


class ExtraMetricOutput:
    """
    This is a base class for calculating and adding auxilary information to the metrics.
    For example, we might want to calculate some very specific samples which can be used to calculate some metrics.
    """

    def __call__(
        self, pl_module: BaseLightningModule, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> TensorDict: ...
    def to(self, device: torch.device) -> None: ...


class MetricsCallback(Callback):
    def __init__(self, metrics: list[BaseMetric], extras: list[ExtraMetricOutput] = []):
        self.metrics = metrics
        self.extras = extras

    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        for metric in self.metrics:
            metric.to(pl_module.device)

        for extra in self.extras:
            extra.to(pl_module.device)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        extras = {}

        for extra in self.extras:
            extra_outputs = extra(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx)
            assert extras.keys().isdisjoint(
                extra_outputs
            ), f"Duplicate extra output keys: {extras.keys() & extra_outputs.keys()}"
            extras.update(extra_outputs)

        for metric in self.metrics:
            metric.add(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx, extras=extras)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        for metric in self.metrics:
            metric_value = metric.compute()
            if metric_value is not None:
                pl_module.log(f"metrics/val/{metric.name()}", metric_value)
            metric.reset()
