from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from torch import Tensor, IntTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.networks import BaseEncoderDecoder
from typing import Any
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import mse_loss
from tqdm import tqdm
from .mixins import EncoderDecoderMixin
from src.lightning_modules.schedulers import FMScheduler

class FM(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        model : torch.nn.Module,
        encoder_decoder : BaseEncoderDecoder | None = None,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | str] | None = None,
        added_noise : float = 0.0,
        latent_std : float = 1.0,
        **kwargs : dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder'])
        
        self.model = model
        self.added_noise = added_noise
        self.latent_std = latent_std
        self.scheduler = FMScheduler(**kwargs)
        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        
    def forward(self, x : Tensor, timesteps : IntTensor) -> Tensor:
        return self.model(x, timesteps)
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)
    
    def common_step(self, x_encoded : Tensor) -> Tensor:
        xt, timesteps, target = self.scheduler.sample_batch(x_encoded)
        model_output = self(xt, timesteps)
        loss = mse_loss(model_output, target)
        return loss
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        x_encoded = self.encode(batch, add_noise=True)
        loss = self.common_step(x_encoded)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        torch.manual_seed(0)
        x_encoded = self.encode(batch)
        loss = self.common_step(x_encoded)
        self.log('val_loss', loss)
        return loss

    @torch.no_grad()
    def sample(self, noise : Tensor, return_trajectory : bool = False, show_progress : bool = False) -> Tensor:
        self.eval()
        xt = noise
        trajectory = [xt]
        for t in tqdm(reversed(self.scheduler.timesteps), desc='Sampling', disable=not show_progress):
            model_output = self(xt, t)
            xt = self.scheduler.step(xt, t, model_output)
            trajectory.append(xt)
            
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory if return_trajectory else trajectory[-1]
        
    def configure_optimizers(self):
        optim = self.partial_optimizer(self.model.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }