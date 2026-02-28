from pytorch_lightning import Callback
from typing import Optional
from src.lightning_modules import BaseLightningModule
import pytorch_lightning as pl
from src import Batch, StepOutput
import time
import logging
import torch
import wandb

logger = logging.getLogger(__name__)


class StopTrainingCallback(Callback):
    """
    This callback is used for stopping the training if some constraints are violated,
    for example if the number of parameters in the model exceeds a certain threshold.
    The callback then stop training and logs some value.
    This callback is usuful for doing Bayesian optimzation; this way, we can
    limit the search space of the hyperparameters and stop training if the model is too large,
    which is a common constraint when doing hyperparameter optimization.

    Args:
        key (str): The key to log when stopping the training.
        value (float): The value to log when stopping the training.
        max_num_params (Optional[int]): The maximum number of parameters allowed in the model. If None, this constraint is not applied.
        max_batch_time (Optional[float]): The maximum time (in seconds) allowed for processing a train batch. If None, this constraint is not applied.
        catch_oom (bool): Whether to catch CUDA out of memory errors and stop training if they occur. Default is True.
    """

    def __init__(
        self,
        key: str,
        value: float,
        max_num_params: Optional[int] = None,
        max_batch_time: Optional[float] = None,
        catch_oom: bool = True,
    ):
        self.key = key
        self.value = value
        self.max_num_params = max_num_params
        self.max_batch_time = max_batch_time
        self.catch_oom = catch_oom

    def _stop_training(self, reason: str):
        wandb.log({self.key: self.value})
        logger.info(f"Stopping training: {reason}.")
        wandb.finish()
        raise KeyboardInterrupt(f"Training stopped by StopTrainingCallback: {reason}")

    def on_train_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        if self.max_num_params is not None:
            num_params = sum(p.numel() for p in pl_module.parameters())
            if num_params > self.max_num_params:
                self._stop_training(
                    reason=f"number of parameters in the model ({num_params:,}) exceeds the maximum allowed ({self.max_num_params:,})",
                )

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: BaseLightningModule, batch: Batch, batch_idx: int
    ) -> None:
        # measure the time it takes to process a train batch
        # measure on the second batch, to avoid measuring the time it takes to load the data
        if self.max_batch_time is not None and batch_idx == 1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # wait for all CUDA operations to finish before measuring time

            self.batch_start_time = time.time()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        if self.max_batch_time is not None and batch_idx == 1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            batch_time = time.time() - self.batch_start_time
            if batch_time > self.max_batch_time:
                self._stop_training(
                    reason=f"the time it takes to process a train batch ({batch_time:.2f} seconds) exceeds the maximum allowed ({self.max_batch_time:.2f} seconds)."
                )

    @staticmethod
    def _is_oom(exception: BaseException) -> bool:
        if isinstance(exception, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(exception, RuntimeError) and "out of memory" in str(exception).lower():
            return True

        return False

    def on_exception(self, trainer: pl.Trainer, pl_module: BaseLightningModule, exception: BaseException) -> None:
        if self.catch_oom and self._is_oom(exception):
            self._stop_training(reason="CUDA out of memory error occurred.")
