import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer
from lightning.pytorch.utilities import grad_norm

class LogGradsCallback(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 1,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        
    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer) -> None:
        if trainer.global_step % self.log_every_n_steps == 0:
            norms = grad_norm(pl_module, norm_type=2)
            pl_module.log_dict(norms)