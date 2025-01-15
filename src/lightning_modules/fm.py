from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.lightning_modules.utils import get_encoder_decoder

class FM(BaseLightningModule):
    def __init__(
        self,
        model : torch.nn.Module,
        encoder_decoder_id : str,
        optimizer : Optimizer,
        scheduler : str = 'ddpm',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'optimizer'])
        
        self.model = model
        self.partial_optimizer = optimizer
        match scheduler:
            case 'ddim':
                self.scheduler = DDIMScheduler(**kwargs)
            case 'ddpm':
                self.scheduler = DDPMScheduler(**kwargs)
            case _:
                raise ValueError(f"Unknown scheduler: {scheduler}")
                
        self.encoder, self.decoder = get_encoder_decoder(encoder_decoder_id)
        self.mse = torch.nn.MSELoss()
    
    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        return self.model(x, timesteps)
    
    @property
    def num_train_timesteps(self) -> int:
        return self.scheduler.timesteps[0].long().item()
    
    @torch.no_grad()
    def encode(self, x : Tensor) -> Tensor:
        return self.encoder(x)
    
    @torch.no_grad()
    def decode(self, x : Tensor) -> Tensor:
        return self.decoder(x)
        
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
    def sample(self, noise : Tensor, num_inference_steps : int | None = None) -> Tensor:
        self.eval()
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
        scheduler = ReduceLROnPlateau(optim, mode = 'min', factor=0.5, patience=5)
        return {
            'optimizer': optim,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }