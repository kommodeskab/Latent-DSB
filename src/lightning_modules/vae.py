import pytorch_lightning as pl
from src.lightning_modules.baselightningmodule import BaseLightningModule
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor
import torch
from torch.optim import Optimizer
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LRScheduler
from typing import Any

class VAE(BaseLightningModule):
    def __init__(
        self,
        vae : AutoencoderKL,
        beta : float = 1.0,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | Any] | None = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.beta = beta
        self.vae = vae
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        
    def encode(self, x : Tensor) -> Tensor:
        return self.vae.encode(x).latent_dist.mean
        
    def decode(self, z : Tensor) -> Tensor:
        return self.vae.decode(z).sample
    
    @torch.no_grad()
    def encode_decode(self, x : Tensor) -> Tensor:
        return self.decode(self.encode(x))
    
    @staticmethod
    def kl_divergence(mean : Tensor, logvar : Tensor) -> Tensor:
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    @staticmethod
    def reconstruction_loss(reconstruction : Tensor, target : Tensor) -> Tensor:
        return mse_loss(reconstruction, target, reduction = 'sum')

    def common_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        batch_size = batch.size(0)
        dist : DiagonalGaussianDistribution = self.vae.encode(batch).latent_dist
        mean, logvar = dist.mean, dist.logvar
        sampled_z = dist.sample()
        reconstruction = self.vae.decode(sampled_z).sample
        kl = self.kl_divergence(mean, logvar) / batch_size
        recon_loss = self.reconstruction_loss(reconstruction, batch) / batch_size
        loss = recon_loss + self.beta * kl
        
        return {
            'loss': loss,
            'kl': kl,
            'recon_loss': recon_loss
        }
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        loss = self.common_step(batch, batch_idx)
        loss = self._convert_dict_losses(loss, prefix = "train")
        self.log_dict(loss)
        return loss['train/loss']
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        loss = self.common_step(batch, batch_idx)
        loss = self._convert_dict_losses(loss, prefix = "val")
        self.log_dict(loss)
        return loss['val/loss']
    
    def configure_optimizers(self):
        optim = self.partial_optimizer(self.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }