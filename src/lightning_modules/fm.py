from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.networks import BaseEncoderDecoder
from typing import Any
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import mse_loss

class FM(BaseLightningModule):
    def __init__(
        self,
        model : torch.nn.Module,
        scheduler : DDIMScheduler | DDPMScheduler,
        encoder_decoder : BaseEncoderDecoder | None = None,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | str] | None = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder'])
        
        self.model = model
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        self.scheduler = scheduler
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps)
        self.encoder_decoder = encoder_decoder
    
    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        return self.model(x, timesteps)
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)

    @torch.no_grad()
    def encode(self, x : Tensor):
        return self.encoder_decoder.encode(x)
    
    @torch.no_grad()
    def decode(self, x : Tensor):
        return self.encoder_decoder.decode(x)
        
    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        return self.model(x, timesteps)
    
    def sample_timesteps(self, batch_size : int) -> Tensor:
        timesteps = self.scheduler.timesteps
        random_indices = torch.randint(0, len(timesteps), (batch_size,))
        return timesteps[random_indices].to(self.device)
    
    def t_to_timesteps(self, t : float, batch_size : int) -> Tensor:
        return torch.full((batch_size,), t).to(self.device)
    
    @property
    def prediction_type(self) -> str:
        return self.scheduler.config.prediction_type
    
    def common_step(self, batch : Tensor) -> Tensor:
        batch_size = batch.size(0)
        x0 = self.encode(batch)
        timesteps = self.sample_timesteps(batch_size)
        noise = torch.randn_like(x0).to(self.device)
        xt = self.scheduler.add_noise(x0, noise, timesteps)
        
        model_output = self(xt, timesteps)
        
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x0
            
        loss = mse_loss(target, model_output)
        return loss
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        loss = self.common_step(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        torch.manual_seed(0)
        loss = self.common_step(batch)
        self.log('val_loss', loss)
        return loss
    
    @torch.no_grad()
    def sample(self, noise : Tensor, return_trajectory : bool = False) -> Tensor:
        self.eval()
        batch_size = noise.size(0)
        xt = noise
        trajectory = [xt]
        for t in self.scheduler.timesteps:
            model_output = self(xt, t)
            xt = self.scheduler.step(model_output, t, xt, eta=1.0).prev_sample
            trajectory.append(xt)
            
        trajectory = torch.stack(trajectory, dim=0)
        self.train()
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