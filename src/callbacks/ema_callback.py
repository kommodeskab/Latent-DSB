from torch_ema import ExponentialMovingAverage
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer
import logging
from src.lightning_modules import BaseLightningModule

logger = logging.getLogger(__name__)


class EMACallback(Callback):
    """
    Exponential Moving Average (EMA) Callback for PyTorch Lightning.
    This callback maintains an exponential moving average of the model parameters during training.
    The EMA weights are used during validation to improve performance and stability.
    The EMA state is saved and loaded with the model checkpoint.

    Args:
        decay (float): The decay rate for the EMA. Default is 0.999.
    """

    def __init__(
        self,
        decay: float = 0.999,
    ):
        super().__init__()
        self.decay = decay

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        logger.info("Initializing EMA...")
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ema.to(pl_module.device)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ema.to(pl_module.device)

    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: BaseLightningModule, optimizer: Optimizer):
        self.ema.update()

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: BaseLightningModule, checkpoint: dict):
        logger.info("Saving EMA state dict to checkpoint.")
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: BaseLightningModule, checkpoint: dict):
        if "ema_state_dict" in checkpoint:
            logger.info("Loading EMA state dict from checkpoint.")
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def on_validation_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule):
        logger.info("Applying EMA weights for validation.")
        self.ema.store()
        self.ema.copy_to()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule):
        logger.info("Restoring original weights after validation.")
        self.ema.restore()
