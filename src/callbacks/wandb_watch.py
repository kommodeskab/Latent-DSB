from pytorch_lightning import Callback
import pytorch_lightning as pl
from src.lightning_modules.baselightningmodule import BaseLightningModule


class WandbWatchCallback(Callback):
    """
    A very simple callback that uses wandb.watch to log the model graph and parameters at the start of training.
    """

    def __init__(
        self,
        log: str = "all",
        log_frequency: int = 1000,
        log_graph: bool = True,
    ):
        self._log = log  # not to overwrite existing 'log' method
        self.log_frequency = log_frequency
        self.log_graph = log_graph

    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        pl_module.logger.watch(pl_module, log=self._log, log_graph=self.log_graph, log_freq=self.log_frequency)
