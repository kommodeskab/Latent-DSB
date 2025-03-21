from src.lightning_modules.schedulers import DSBScheduler
from src.lightning_modules.baselightningmodule import BaseLightningModule
from src.networks.encoders import BaseEncoderDecoder
from torch.optim import Optimizer
from torch.nn import Module
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import mse_loss
from .mixins import EncoderDecoderMixin
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import LRScheduler
from typing import Literal
from torch import Tensor, IntTensor
import torch
from src.lightning_modules.utils import sort_dict_by_model
from tqdm import tqdm
from pytorch_lightning.utilities.seed import isolate_rng

class InitDSB(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        num_timesteps : int = 100,
        gamma_min : float = 1e-3,
        gamma_max : float = 1e-2,
        added_noise : float = 0.0,
        target : Literal['flow', 'terminal'] = 'flow',
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | str] | None = None,
        ema_decay : float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder'])
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        self.added_noise = added_noise
        self.scheduler = DSBScheduler(num_timesteps, gamma_min, gamma_max, target)
        
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        
    def on_fit_start(self):
        self.ema.to(self.device)
        
    def forward(self, x : Tensor, timesteps : IntTensor) -> Tensor:
        return self.model(x, timesteps)
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)
        
    def common_step(self, x0 : Tensor, x1 : Tensor) -> Tensor:
        trajectory = self.scheduler.deterministic_sample(x0, x1, return_trajectory=True, noise='training')
        xt, timesteps, target = self.scheduler.sample_batch(trajectory)
        model_output = self(xt, timesteps)
        loss = mse_loss(model_output, target)
        return loss
    
    def training_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        x0, x1 = batch
        x0, x1 = self.encode(x0, add_noise=True), self.encode(x1, add_noise=True)
        loss = self.common_step(x0, x1)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        with self.fix_validation_seed():
            x0, x1 = batch
            x0, x1 = self.encode(x0, add_noise=False), self.encode(x1, add_noise=False)
            with self.ema.average_parameters():
                loss = self.common_step(x0, x1)
                
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
        noise : Literal['inference', 'none'] = 'inference'
    ) -> Tensor:
        self.eval()
        batch_size = x_start.size(0)
        xt = x_start
        trajectory = [xt]
        for t in tqdm(reversed(self.scheduler.timesteps), desc='Sampling', disable=not show_progress):
            timesteps = torch.full((batch_size,), t, dtype=torch.int64, device=xt.device)
            with self.ema.average_parameters():
                model_output = self(xt, timesteps)
            xt = self.scheduler.step(xt, t, model_output, noise)
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