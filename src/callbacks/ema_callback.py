from torch_ema import ExponentialMovingAverage
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer
import logging

logger = logging.getLogger(__name__)

class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.999,
    ):
        super().__init__()
        self.decay = decay
        
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info("Initializing EMA...")
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)
        self.ema.to(pl_module.device)
        pl_module.ema = self.ema
        
    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningDataModule, optimizer: Optimizer) -> None:
        self.ema.update()
        
    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        logger.info("Saving EMA state dict to checkpoint.")
        with self.ema.average_parameters(): 
            # Save the model parameters with EMA weights
            checkpoint['state_dict'] = pl_module.state_dict()
                                    
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info("Applying EMA weights for validation.")
        self.ema.store()
        self.ema.copy_to()
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.ema.collected_params is None:
            logger.info("Restoring original weights after validation.")
            self.ema.restore()        
    
