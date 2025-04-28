from src.lightning_modules.schedulers import DSBScheduler, NOISE_TYPES, TARGETS
from src.lightning_modules.baselightningmodule import BaseLightningModule
from src.networks.encoders import BaseEncoderDecoder
from torch.optim import Optimizer
from torch.nn import Module
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import mse_loss
from .mixins import EncoderDecoderMixin
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor, IntTensor
import torch
from tqdm import tqdm
from time import time

class InitDSB(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        num_timesteps : int = 100,
        target : TARGETS = 'flow',
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | str] | None = None,
        ema_decay : float = 0.999,
        gamma_min : float | None = None,
        gamma_max : float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder', 'optimizer', 'lr_scheduler'])
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay, use_num_updates=True)
        self.scheduler = DSBScheduler(num_timesteps, gamma_min, gamma_max, target)
        
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
                
    def on_fit_start(self):
        self.ema.to(self.device)
        self.encoder_decoder.to(self.device)
        
    def forward(self, x : Tensor, timesteps : IntTensor) -> Tensor:
        return self.model(x, timesteps)
    
    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 100 == 0:   
            norm = grad_norm(self.model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.log("norm", norm, prog_bar=True)
        
    def common_step(self, x0 : Tensor, x1 : Tensor) -> Tensor:
        xt, timesteps, target = self.scheduler.sample_init_batch(x0, x1)
        model_output = self(xt, timesteps)
        loss = mse_loss(model_output, target)
        return loss
    
    def training_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        x0_encoded, x1_encoded = self.encode_batch(batch)
        loss = self.common_step(x0_encoded, x1_encoded)
        self.log_dict({
            'train_loss': loss,
        }, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        with self.fix_validation_seed():
            x0_encoded, x1_encoded = self.encode_batch(batch)
            with self.ema.average_parameters():
                loss = self.common_step(x0_encoded, x1_encoded)
                
            self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def state_dict(self):
        with self.ema.average_parameters():
            ema_state_dict = super().state_dict().copy()
            
        return ema_state_dict
    
    def on_before_zero_grad(self, optimizer):
        self.ema.update()
        
    @torch.no_grad()
    def sample(
        self, 
        x_start : Tensor, 
        return_trajectory : bool = False, 
        show_progress : bool = False, 
        noise : NOISE_TYPES = 'inference'
    ) -> Tensor:
        self.model.eval()
        batch_size = x_start.size(0)
        xt = x_start.clone()
        trajectory = [xt]
        for k in tqdm(reversed(self.scheduler.timesteps), desc='Sampling', disable=not show_progress, leave=False):
            timesteps = torch.full((batch_size,), k, dtype=torch.int64, device=xt.device)
            with self.ema.average_parameters():
                model_output = self(xt, timesteps)
            xt = self.scheduler.step(xt, k, model_output, noise)
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