from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from src.networks import BaseEncoderDecoder

class FM(BaseLightningModule):
    def __init__(
        self,
        model : torch.nn.Module,
        scheduler : DDIMScheduler | DDPMScheduler,
        encoder_decoder : BaseEncoderDecoder,
        optimizer : Optimizer | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'optimizer', 'encoder_decoder'])
        
        self.model = model
        self.partial_optimizer = optimizer
        self.scheduler = scheduler
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.encoder_decoder = encoder_decoder
        self.mse = torch.nn.MSELoss()
    
    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        return self.model(x, timesteps)

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
    
    def common_step(self, batch : Tensor) -> Tensor:
        batch_size = batch.size(0)
        x0 = self.encode(batch)
        timesteps = self.sample_timesteps(batch_size)
        noise = torch.randn_like(x0).to(self.device)
        xt = self.scheduler.add_noise(x0, noise, timesteps)
        model_output = self(xt, timesteps)
        loss = self.mse(noise, model_output)
        return loss
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        loss = self.common_step(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        loss = self.common_step(batch)
        self.log('val_loss', loss)
        return loss
    
    @torch.no_grad()
    def sample(self, noise : Tensor, return_trajectory : bool = False, num_steps : int | None = None) -> Tensor:
        num_timesteps = num_steps or self.num_train_timesteps
        self.scheduler.set_timesteps(num_timesteps, self.device)
        self.eval()
        batch_size = noise.size(0)
        xt = noise
        trajectory = [xt]
        for t in self.scheduler.timesteps:
            timesteps = self.t_to_timesteps(t, batch_size)
            model_output = self(xt, timesteps)
            xt = self.scheduler.step(model_output, t, xt).prev_sample
            trajectory.append(xt)
            
        trajectory = torch.stack(trajectory, dim=0)
        self.train()
        return trajectory if return_trajectory else trajectory[-1]
        
    def configure_optimizers(self):
        optim = self.partial_optimizer(self.model.parameters())
        scheduler = CosineAnnealingWarmRestarts(optim, T_0 = 100, T_mult=2)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'monitor': 'val_loss'
            }
        }