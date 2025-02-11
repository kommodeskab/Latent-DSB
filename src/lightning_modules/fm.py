from src.lightning_modules.baselightningmodule import BaseLightningModule
import torch
from torch import Tensor, IntTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.networks import BaseEncoderDecoder
from typing import Any
from pytorch_lightning.utilities import grad_norm
from torch.nn.functional import mse_loss
from src.lightning_modules.utils import GammaScheduler
from typing import Tuple
from tqdm import tqdm

class FMScheduler:
    def __init__(self, num_timesteps : int = 1000, gamma_frac: float = 1.0):
        self.num_train_timesteps = num_timesteps
        self.eps_min = 1e-4
        self.set_timesteps(num_timesteps, gamma_frac)
    
    def set_timesteps(self, num_timesteps : int, gamma_frac: float = 1.0) -> None:
        scheduler = GammaScheduler(gamma_frac, 1, num_timesteps, 1)
        sigmas = scheduler.gammas_bar
        timesteps = (sigmas * self.num_train_timesteps).int()
        timesteps = timesteps[1:]
        timesteps = torch.clamp(timesteps, min=1)
        sigmas[-1] = 1 - self.eps_min
        self.timesteps = timesteps
        self.sigmas = sigmas
        
    def sample_timesteps(self, batch_size : int) -> Tensor:
        random_indices = torch.randint(0, len(self.timesteps), (batch_size,))
        return self.timesteps[random_indices]
    
    def sample_batch(self, batch : Tensor) -> Tuple[Tensor, IntTensor, Tensor]:
        batch_size = batch.size(0)
        device = batch.device
        sigmas = self.sigmas.type_as(batch)
        sigmas = self.to_dim(sigmas, batch.dim())
        timesteps = self.sample_timesteps(batch_size).type_as(batch)
        sigmas = sigmas[timesteps]
        noise = torch.randn_like(batch)
        target = noise - batch
        xt = (1 - sigmas) * batch + sigmas * noise
        return xt, timesteps, target
    
    def step(self, xt_plus_1 : Tensor, t_plus_1 : int, model_output : Tensor) -> Tensor:
        """
        Predict x_t | x_{t + 1}
        """
        assert t_plus_1 > 0, "Timestep must be greater than 0"
        sigmas = self.sigmas.type_as(xt_plus_1)
        delta_t = sigmas[t_plus_1 - 1] - sigmas[t_plus_1]
        return xt_plus_1 + delta_t * model_output
            
    def to_dim(self, x : torch.Tensor, dim : int) -> torch.Tensor:
        while x.dim() < dim:
            x = x.unsqueeze(-1)
        return x

class FM(BaseLightningModule):
    def __init__(
        self,
        model : torch.nn.Module,
        encoder_decoder : BaseEncoderDecoder | None = None,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, LRScheduler | str] | None = None,
        added_noise : float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder'])
        
        self.model = model
        self.added_noise = added_noise
        self.scheduler = FMScheduler()
        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
            
    def forward(self, x : Tensor, timesteps : IntTensor) -> Tensor:
        return self.model(x, timesteps)
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)

    @torch.no_grad()
    def encode(self, x : Tensor, add_noise : bool = False) -> Tensor:
        encoded = self.encoder_decoder.encode(x)
        if add_noise:
            encoded += self.added_noise * torch.randn_like(encoded)
        return encoded
    
    @torch.no_grad()
    def decode(self, x : Tensor):
        return self.encoder_decoder.decode(x)
    
    def common_step(self, x_encoded : Tensor) -> Tensor:
        xt, timesteps, target = self.scheduler.sample_batch(x_encoded)
        model_output = self(xt, timesteps)
        loss = mse_loss(model_output, target)
        return loss
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        x_encoded = self.encode(batch, add_noise=True)
        loss = self.common_step(x_encoded)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        torch.manual_seed(0)
        x_encoded = self.encode(batch)
        loss = self.common_step(x_encoded)
        self.log('val_loss', loss)
        return loss

    @torch.no_grad()
    def sample(self, noise : Tensor, return_trajectory : bool = False) -> Tensor:
        self.eval()
        xt = noise
        trajectory = [xt]
        for t in reversed(self.scheduler.timesteps):
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