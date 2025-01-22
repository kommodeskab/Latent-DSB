import pytorch_lightning as pl
from src.lightning_modules.baselightningmodule import BaseLightningModule
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor
import torch
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class VAE(BaseLightningModule):
    def __init__(
        self,
        beta : float = 1.0,
        optimizer : Optimizer | None = None,
    ):
        super().__init__()
        self.beta = beta
        self.vae : AutoencoderKL = AutoencoderKL.from_pretrained(f"stable-diffusion-vae/3.5")
        self.partial_optimizer = optimizer
        
    def encode(self, x : Tensor) -> tuple[Tensor, Tensor]:
        dist : DiagonalGaussianDistribution = self.vae.encode(x).latent_dist
        return dist.mean, dist.logvar
    
    def decode(self, z : Tensor) -> Tensor:
        return self.vae.decode(z).sample
    
    @torch.no_grad()
    def encode_decode(self, x : Tensor) -> Tensor:
        mean, _ = self.encode(x)
        return self.decode(mean)
    
    def reparameterize(self, mean : Tensor, logvar : Tensor) -> Tensor:
        std = 0.5 * logvar.exp()
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def loss_function(self, recon_x : Tensor, x : Tensor, mean : Tensor, logvar : Tensor) -> Tensor:
        # calculate average mse loss and kl divergence
        batch_size = x.size(0)
        reconstruction_loss = mse_loss(recon_x, x, reduction='sum')
        reconstruction_loss /= batch_size
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_divergence /= batch_size
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'kl_divergence': kl_divergence,
            'loss': reconstruction_loss + self.beta * kl_divergence
        } 

    def common_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        mean, logvar = self.encode(batch)
        z = self.reparameterize(mean, logvar)
        recon_batch = self.decode(z)
        loss = self.loss_function(recon_batch, batch, mean, logvar)
        return loss
    
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
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'val_loss',
                'frequency': 1
            }
        }