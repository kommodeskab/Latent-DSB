import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from typing import Dict

TensorDict = Dict[str, Tensor]

class LogLossCallback(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs : TensorDict, batch, batch_idx):
        outputs = {f"train_{k}": v for k, v in outputs.items() if v.numel() == 1}
        pl_module.log_dict(outputs, prog_bar=True)
        
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs : TensorDict, batch, batch_idx, dataloader_idx=0):
        outputs = {f"val_{k}": v for k, v in outputs.items() if v.numel() == 1}
        pl_module.log_dict(outputs, prog_bar=True)
        
    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs : TensorDict, batch, batch_idx, dataloader_idx=0):
        outputs = {f"test_{k}": v for k, v in outputs.items() if v.numel() == 1}
        pl_module.log_dict(outputs, prog_bar=True)
