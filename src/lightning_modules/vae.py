from torch import Tensor
import torch
import torch.nn as nn
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.optim import Optimizer
from typing import Optional
from functools import partial
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict
from src.losses import BaseLoss
from src.networks import VAENetwork

TensorDict = Dict[str, Tensor]

class VAEModel(BaseLightningModule):
    def __init__(
        self,
        model: VAENetwork,
        loss_fn: BaseLoss,
        optimizer: Optional[partial[Optimizer]] = None,
        lr_scheduler: Optional[dict[str, partial[LRScheduler] | str]] = None,
    ):
        super().__init__(optimizer, lr_scheduler)
        self.model = model
        self.loss_fn = loss_fn
        
    def common_step(self, x : Tensor, batch_idx : int) -> TensorDict:
        q_z = self.model.encode(x)
        z_sample = q_z.rsample()
        p_x = self.model.decode(z_sample)
        model_output = {
            'q_z': q_z,
            'p_x': p_x,
            'target': x,
            'model': self,
        }
        loss = self.loss_fn(model_output)
        loss.update(model_output)
        return loss