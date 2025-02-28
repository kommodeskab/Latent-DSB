from src.lightning_modules.schedulers import ReFlowScheduler
from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from torch import Tensor, IntTensor
from .mixins import EncoderDecoderMixin
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.networks import BaseEncoderDecoder
from torch_ema import ExponentialMovingAverage
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import mse_loss
from tqdm import tqdm

class ReFlow(BaseLightningModule, EncoderDecoderMixin):
    def __init__(
        self,
        model : torch.nn.Module,
        training_backward : bool,
        encoder_decoder : BaseEncoderDecoder | None = None,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | str] | None = None,
        added_noise : float = 0.0,
        num_timesteps : int = 100,
        ema_decay : float = 0.999,
    ):
        self.save_hyperparameters(ignore=['model', 'encoder_decoder'])
        
        self.model = model
        self.training_backward = training_backward
        self.added_noise = added_noise
        self.scheduler = ReFlowScheduler(num_timesteps=num_timesteps)
        self.encoder_decoder = encoder_decoder
        self.added_noise = added_noise
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
    
    def on_fit_start(self):
        self.ema.to(self.device)
        
    def forward(self, x : Tensor, timesteps : IntTensor) -> Tensor:
        return self.model(x, timesteps)
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)
        
    def common_step(self, x0 : Tensor, x1 : Tensor) -> Tensor:
        xt, timesteps, target = self.scheduler.sample_batch(x0, x1, self.training_backward)
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
        torch.manual_seed(0)
        x0, x1 = batch
        x0, x1 = self.encode(x0, add_noise=False), self.encode(x1, add_noise=False)
        with self.ema.average_parameters():
            loss = self.common_step(x0, x1)
            
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_before_zero_grad(self, optimizer):
        self.ema.update()
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()
        
    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint['ema'])
        self.ema.copy_to(self.model.parameters())
        print("Loaded EMA weights")
        
    def sample(self, x_start : Tensor, return_trajectory : bool = True, show_progress : bool = False) -> Tensor:
        self.eval()
        batch_size = x_start.size(0)
        xk = x_start
        trajectory = [xk]
        timesteps = self.scheduler.get_timesteps_for_sampling(self.training_backward)
        for k in tqdm(timesteps, desc='Sampling', disable=not show_progress):
            ks = torch.full((batch_size,), k, dtype=torch.int64, device=x_start.device)
            with self.ema.average_parameters():
                model_output = self(xk, ks)
            xt = self.scheduler.step(xk, k, model_output, self.training_backward)
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