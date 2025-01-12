from diffusers import FlowMatchEulerDiscreteScheduler
from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.lightning_modules.utils import get_encoder_decoder

class FM(BaseLightningModule):
    def __init__(
        self,
        model : torch.nn.Module,
        encoder_decoder_id : str,
        optimizer : Optimizer,
        scheduler : LRScheduler,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'op'])
        
        self.model = model
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(**kwargs)
        self.encoder, self.decoder = get_encoder_decoder(encoder_decoder_id)
        self.mse = torch.nn.MSELoss()
        
    @property
    def num_train_timesteps(self) -> int:
        return self.scheduler.timesteps[0].long().item()
        
    def encode(self, x : Tensor) -> Tensor:
        return self.encoder(x)
    
    def decode(self, x : Tensor) -> Tensor:
        return self.decoder(x)
        
    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        return self.model(x, timesteps)
    
    def sample_timesteps(self, batch_size : int) -> Tensor:
        return torch.randint(1, self.num_train_timesteps, (batch_size,)).to(self.device)
    
    def t_to_timesteps(self, t : float, batch_size : int) -> Tensor:
        return torch.tensor([t]*batch_size).to(self.device)
    
    def common_step(self, batch : Tensor) -> Tensor:
        batch_size = batch.size(0)
        x0 = self.encode(batch)
        timesteps = self.sample_timesteps(batch_size)
        noise = torch.randn_like(x0).to(self.device)
        xt = self.scheduler.scale_noise(x0, timesteps, noise)
        model_output = self.model(xt, timesteps)
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
    
    def sample(self, noise : Tensor, num_inference_steps : int | None = None) -> Tensor:
        batch_size = noise.size(0)
        num_inference_steps = num_inference_steps or self.num_train_timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        ts = self.scheduler.timesteps
        xt = noise
        for t in ts:
            timesteps = self.t_to_timesteps(t, batch_size)
            model_output = self(xt, timesteps)
            xt = self.scheduler.step(model_output, t, xt).prev_sample
            
        return xt
        
    def configure_optimizers(self):
        optim = self.partial_optimizer(self.model.parameters())
        scheduler = self.partial_scheduler(optim)
        return {
            'optimizer': optim,
            'lr_scheduler': scheduler,
        }