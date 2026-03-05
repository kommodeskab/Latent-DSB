from pytorch_lightning import Callback
from src import StepOutput, TensorDict, Batch
from src.lightning_modules import BaseLightningModule
import pytorch_lightning as pl
from typing import Literal
from src.callbacks.metrics import BaseMetric
from src.callbacks.extras import ExtraMetricOutput


class MetricsCallback(Callback):
    def __init__(self, metrics: list[BaseMetric], extras: list[ExtraMetricOutput] = []):
        self.metrics = metrics
        self.extras = extras

        # check that none of the metrics have duplicate names
        metric_names = [metric.name() for metric in self.metrics]
        assert len(metric_names) == len(
            set(metric_names)
        ), f"Duplicate metric names: {set([name for name in metric_names if metric_names.count(name) > 1])}"

    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        for metric in self.metrics:
            metric.to(pl_module.device)

        for extra in self.extras:
            extra.to(pl_module.device)

    def _add_extras(
        self, pl_module: BaseLightningModule, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> TensorDict:
        """
        This method calculates the extra outputs for the current batch and returns them as a dictionary.
        The keys of the dictionary are the names of the extra outputs, and the values are the corresponding tensors.
        The extra outputs can then be used in the `add()` method of the metrics to calculate the metric values.
        """
        extras = {}
        for extra in self.extras:
            extra_outputs = extra(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx)
            assert extras.keys().isdisjoint(
                extra_outputs
            ), f"Duplicate extra output keys: {extras.keys() & extra_outputs.keys()}"
            extras.update(extra_outputs)
        return extras

    def _add_metrics(
        self, pl_module: BaseLightningModule, outputs: StepOutput, batch: Batch, batch_idx: int, extras: TensorDict
    ) -> None:
        """
        This method adds the current batch's outputs to the metrics.
        It should be called at the end of each validation batch.
        The metrics will accumulate the necessary information from each batch, and then compute the final metric values at the end of the epoch.
        """
        for metric in self.metrics:
            metric.add(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx, extras=extras)

    def _compute_metrics(self, pl_module: BaseLightningModule, phase: Literal["val", "test"]) -> None:
        """
        This method computes the final metric values and logs them to PyTorch Lightning.
        It should be called at the end of each validation epoch.
        The metric values will be logged under the key `metrics/{phase}/{metric_name}`,
        where `{phase}` is either 'val' or 'test', and `{metric_name}` is the name of the metric as returned by its `name()` method.
        """

        for metric in self.metrics:
            metric_value = metric.compute()
            if metric_value is not None:
                pl_module.log(f"metrics/{phase}/{metric.name()}", metric_value)
            metric.reset()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        extras = self._add_extras(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx)
        self._add_metrics(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx, extras=extras)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        self._compute_metrics(pl_module=pl_module, phase="val")

    def on_test_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        extras = self._add_extras(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx)
        self._add_metrics(pl_module=pl_module, outputs=outputs, batch=batch, batch_idx=batch_idx, extras=extras)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        self._compute_metrics(pl_module=pl_module, phase="test")
