import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch.nn.init as init
import torch
from contextlib import contextmanager
import random
import numpy as np
from torch.optim import Optimizer
from typing import Optional
from functools import partial
from torch.optim.lr_scheduler import LRScheduler
from src.data_modules import BaseDM

class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        optimizer: Optional[partial[Optimizer]] = None,
        lr_scheduler: Optional[dict[str, partial[LRScheduler] | str]] = None,
        ):
        super().__init__()
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        
    @property
    def datamodule(self) -> BaseDM:
        return self.trainer.datamodule
        
    @property
    def logger(self) -> WandbLogger:
        return self.trainer.logger
    
    @contextmanager
    def fix_validation_seed(self):
        cuda = self.device.type == 'cuda'
        
        cpu_rng_state = torch.get_rng_state()
        random_rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        if cuda:
            cuda_rng_state = torch.cuda.get_rng_state()
            
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        
        yield
        
        torch.set_rng_state(cpu_rng_state)
        random.setstate(random_rng_state)
        np.random.set_state(np_rng_state)
        if cuda:
            torch.cuda.set_rng_state(cuda_rng_state)
    
    @staticmethod
    def init_weights(model : nn.Module) -> None:
        """
        Initializes the weights of the forward and backward models  
        using the Kaiming Normal initialization
        """
        @torch.no_grad()
        def initialize(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Apply initialization to both networks
        model.apply(initialize)
        
    def common_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.common_step(batch, batch_idx)
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        with self.fix_validation_seed():
            return self.common_step(batch, batch_idx)
        
    def configure_optimizers(self):
        assert self.partial_optimizer is not None, "Optimizer must be provided during training."
        assert self.partial_lr_scheduler is not None, "Learning rate scheduler must be provided during training."
        
        optim = self.partial_optimizer(self.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }