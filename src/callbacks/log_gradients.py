import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer
from pytorch_lightning.utilities.grads import grad_norm


class LogGradsCallback(Callback):
    """
    Callback to log gradient norms during training.
    Logs the norm of every module in the model as well as the total gradient norm.

    Args:
        log_every_n_steps (int): Frequency of logging gradient norms in terms of training steps. Default is 100.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer
    ) -> None:
        if trainer.global_step % self.log_every_n_steps == 0:
            norms = grad_norm(pl_module, norm_type=2)
            pl_module.log_dict(norms)
