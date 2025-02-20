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
from torch_ema import ExponentialMovingAverage

class FM(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        model : torch.nn.Module,
        encoder_decoder : BaseEncoderDecoder | None = None,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | str] | None = None,
        added_noise : float = 0.0,
        latent_std : float = 1.0,
        num_timesteps : int = 1000,
        **kwargs : Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder'])
        
        self.model = model
        self.added_noise = added_noise
        self.latent_std = latent_std
        self.scheduler = FMScheduler(num_timesteps)
        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        self.ema = None
        
    def on_fit_start(self):
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        return super().on_fit_start()
        
    def forward(self, x : Tensor, timesteps : IntTensor) -> Tensor:
        return self.model(x, timesteps)
    
    def on_save_checkpoint(self, checkpoint):
        self.ema.store()
        self.ema.copy_to()
        return super().on_save_checkpoint(checkpoint)
    
    def on_train_epoch_start(self):
        if self.ema.collected_params is not None:
            self.ema.restore()
        return super().on_train_epoch_start()
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)
    
    def common_step(self, x0 : Tensor, x1 : Tensor) -> Tensor:
        xt, timesteps, target = self.scheduler.sample_batch(x0, x1)
        model_output = self(xt, timesteps)
        loss = mse_loss(model_output, target)
        return loss
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        x0, x1 = batch
        x0, x1 = self.encode(x0, add_noise=True), self.encode(x1, add_noise=True)
        loss = self.common_step(x0, x1)
        self.log('train_loss', loss)
        return loss
    
    def on_before_zero_grad(self, optimizer):
        self.ema.update()
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        torch.manual_seed(0)
        x0, x1 = batch
        x0, x1 = self.encode(x0, add_noise=True), self.encode(x1, add_noise=True)
        with self.ema.average_parameters():
            loss = self.common_step(x0, x1)
        
        self.log('val_loss', loss)
        return loss

    @torch.no_grad()
    def sample(self, x_start : Tensor, return_trajectory : bool = False, show_progress : bool = False) -> Tensor:
        self.eval()
        xt = x_start
        trajectory = [xt]
        for t in tqdm(reversed(self.scheduler.timesteps), desc='Sampling', disable=not show_progress):
            if self.ema is not None:
                with self.ema.average_parameters():
                    model_output = self(xt, t)
            else:
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