import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from src import StepOutput
from src.lightning_modules import BaseLightningModule


class LogLossCallback(Callback):
    """
    Standard callback for logging loss values.
    The callback will log all values that are returned in the `loss_output` of the loss class.
    The callback adds a prefix to the logged name to distinguish "train", "validation" and "test" values.
    """

    def __init__(self):
        super().__init__()

    def log_outputs(self, pl_module: BaseLightningModule, outputs: StepOutput, prefix: str):
        outputs = {k: v for k, v in outputs["loss_output"].items() if isinstance(v, Tensor)}
        outputs = {f"{prefix}_{k}": v for k, v in outputs.items() if v.numel() == 1}
        pl_module.log_dict(outputs, prog_bar=True, sync_dist=(prefix != "train"))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch,
        batch_idx,
    ):
        self.log_outputs(pl_module, outputs, "train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self.log_outputs(pl_module, outputs, "val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self.log_outputs(pl_module, outputs, "test")
