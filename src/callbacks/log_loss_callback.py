import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from src.types import TensorDict

class LogLossCallback(Callback):
    def __init__(self):
        super().__init__()
        
    def log_outputs(self, pl_module: pl.LightningModule, outputs : TensorDict, prefix: str):
        outputs = {k: v for k, v in outputs.items() if isinstance(v, Tensor)}
        outputs = {f"{prefix}_{k}": v for k, v in outputs.items() if v.numel() == 1}
        pl_module.log_dict(outputs, prog_bar=True)
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: TensorDict, batch, batch_idx):
        self.log_outputs(pl_module, outputs, "train")
        
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: TensorDict, batch, batch_idx, dataloader_idx=0):
        self.log_outputs(pl_module, outputs, "val")
        
    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: TensorDict, batch, batch_idx, dataloader_idx=0):
        self.log_outputs(pl_module, outputs, "test")
