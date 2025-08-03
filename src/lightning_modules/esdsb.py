from time import time
from torch import Tensor
import torch
from typing import Literal, get_args, Any
import math
import numpy as np
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.functional import mse_loss
from tqdm import tqdm
from lightning.pytorch.utilities import grad_norm
from torch_ema import ExponentialMovingAverage
from functools import partial
from src.lightning_modules.mixins import EncoderDecoderMixin
from src.networks.encoders import BaseEncoderDecoder

DIRECTIONS = Literal['forward', 'backward']
SCHEDULER_TYPES = Literal['linear', 'cosine']

class DSBScheduler:
    def __init__(
        self, 
        deterministic : bool = False, 
        ):
        self.deterministic = deterministic

    def sample_noise(self, shape : tuple[int, ...], device: str | torch.device) -> Tensor:
        """Sample noise from a normal distribution.

        Args:
            shape (tuple[int, ...]): The shape of the noise tensor.
            device (str | torch.device): The device to create the noise tensor on.

        Returns:
            Tensor: The sampled noise tensor.
        """
        if self.deterministic:
            return torch.zeros(shape, device=device)
        
        return torch.randn(shape, device=device)
    
    def get_conditional(self, direction : DIRECTIONS | list[DIRECTIONS], device: str | torch.device, batch_size : int | None = None) -> Tensor:
        """Get the conditional mask for the given direction. 1 for 'forward', 0 for 'backward'.

        Args:
            direction (DIRECTIONS | list[DIRECTIONS]): The direction(s) to condition on.
            device (str | torch.device): The device to create the mask on.
            batch_size (int | None, optional): The batch size. Defaults to None.

        Returns:
            Tensor: The conditional mask.
        """
        if isinstance(direction, str):
            direction = [direction] * batch_size
            
        mask = np.array(direction) == 'forward'
        mask = torch.from_numpy(mask).to(device, dtype=torch.long)
        return mask

    def sample_xt(self, x0 : Tensor, x1 : Tensor, timesteps : Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Sample the latent variable at timestep t.

        Args:
            x0 (Tensor): The initial state tensor (p_0).
            x1 (Tensor): The final state tensor (p_1).
            timesteps (Tensor | None, optional): The timesteps to sample from. Defaults to None.

        Returns:
            tuple[Tensor, Tensor]: The sampled latent variable and the timesteps.
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=device)

        t = self.to_dim(timesteps, x0.dim())
        
        xt_mean = (1 - t) * x0 + t * x1
        xt_var = 2 * t * (1 - t)
        noise = self.sample_noise(xt_mean.shape, device)
        xt = xt_mean + xt_var ** 0.5 * noise
        
        return xt, timesteps
    
    def sample_training_batch(self, x0 : Tensor, x1 : Tensor, direction : list[DIRECTIONS] | DIRECTIONS, timesteps : Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:       
        """
        Sample a training batch from the given initial and final states.
        
        Args:
            x0 (Tensor): The initial state tensor (p_0).
            x1 (Tensor): The final state tensor (p_1).
            direction (list[DIRECTIONS] | DIRECTIONS): The direction(s) to sample from
            timesteps (Tensor | None, optional): The timesteps to sample from. Defaults to None.
        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: The sampled latent variable (xt), timesteps, conditional mask, and flow.
        """ 
        batch_size = x0.shape[0]
        
        if isinstance(direction, str):
            direction = [direction] * batch_size
        
        # assert that direction contains only 'forward' or 'backward'
        if not all(d in get_args(DIRECTIONS) for d in direction):
            raise ValueError(f"Invalid direction(s) in {direction}. Use either: {get_args(DIRECTIONS)}")    
        
        conditional = self.get_conditional(direction, x0.device, batch_size)
        mask = conditional.bool() # true for 'forward', false for 'backward'
        
        xt, timesteps = self.sample_xt(x0, x1, timesteps)
        t = self.to_dim(timesteps, x0.dim())
        
        flow = torch.empty_like(xt)
        flow[mask] = (x1[mask] - xt[mask]) / (1 - t[mask]).clamp(min=1e-5)
        flow[~mask] = (x0[~mask] - xt[~mask]) / t[~mask].clamp(min=1e-5)
        
        return xt, timesteps, conditional, flow
        
    def get_dummy_trajectory(self, x0 : Tensor, x1 : Tensor, trajectory_length : int) -> Tensor:
        """Generate a dummy trajectory by sampling from the initial and final states.

        Args:
            x0 (Tensor): The initial state tensor (p_0).
            x1 (Tensor): The final state tensor (p_1).
            trajectory_length (int): The length of the trajectory to generate.

        Returns:
            Tensor: The generated dummy trajectory.
        """
        dim = x0.dim()
        x0, x1 = x0.unsqueeze(0), x1.unsqueeze(0)
        x0 = x0.repeat(trajectory_length, *[1] * dim)
        x1 = x1.repeat(trajectory_length, *[1] * dim)
        timesteps = torch.linspace(0, 1, trajectory_length, device=x0.device)
        xt, _ = self.sample_xt(x0, x1, timesteps)
        return xt
    
    def step(self, xt : Tensor, flow : Tensor, tk_plus_one : float, tk : float, direction : DIRECTIONS) -> Tensor:
        assert tk_plus_one > tk, f"tk_plus_one ({tk_plus_one}) must be greater than tk ({tk})."
        
        delta_t = tk_plus_one - tk # the step size
        xnext_mean = xt + delta_t * flow
        
        if direction == 'backward':
            xnext_var = 2 * delta_t * tk / tk_plus_one
        elif direction == 'forward':
            xnext_var = 2 * delta_t * (1 - tk_plus_one) / (1 - tk)
        
        noise = self.sample_noise(xnext_mean.shape, xt.device)
        
        xnext = xnext_mean + xnext_var ** 0.5 * noise
        return xnext
    
    def get_timeschedule(self, num_steps : int, scheduler_type : SCHEDULER_TYPES, direction : DIRECTIONS) -> list[tuple[float, float]]:  
        if scheduler_type == 'linear':
            t = torch.linspace(0, 1, num_steps + 1)
        elif scheduler_type == 'cosine':
            t = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, num_steps + 1)))
        else:
            raise ValueError(f"Unsupported time schedule type. Use either: {get_args(SCHEDULER_TYPES)}")
        
        t = t.tolist()
        timeschedule = [(t[i + 1], t[i]) for i in range(num_steps)]
        
        if direction == 'backward':
            timeschedule = timeschedule[::-1]
        
        return timeschedule
            
    @staticmethod
    def to_dim(x : Tensor, dim : int) -> Tensor:
        while x.dim() < dim:
            x = x.unsqueeze(-1)
        return x
    
class ESDSB(BaseLightningModule, EncoderDecoderMixin):    
    def __init__(
        self,
        model : Module,
        encoder_decoder : BaseEncoderDecoder,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, partial[LRScheduler] | str] | None = None,
        ema_decay : float = 0.999,
        **scheduler_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'encoder_decoder', 'optimizer', 'lr_scheduler'])
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        self.scheduler = DSBScheduler(**scheduler_kwargs)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        self.stop_epoch = False
            
    def on_before_optimizer_step(self, optimizer : Optimizer) -> None:
        if self.global_step % 500 == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def on_before_zero_grad(self, optimizer : Optimizer) -> None:
        self.ema.update()

    def state_dict(self) -> dict:
        with self.ema.average_parameters():
            state_dict = super().state_dict()
        # dont save encoder_decoder weights since they are frozen during training
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('encoder_decoder.')}
        return state_dict
    
    def load_state_dict(self, state_dict : dict[str, Any], strict = True, assign = False):
        # add encoder_decoder weights back into the state_dict
        encoder_state_dict = self.encoder_decoder.state_dict()
        encoder_state_dict = {f'encoder_decoder.{k}': v for k, v in encoder_state_dict.items()}
        state_dict.update(encoder_state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def to(self, device : torch.device):
        # ema parameters have to be manually moved to the device
        self.ema.to(device)
        return super().to(device)
        
    def forward(self, x : Tensor, timesteps : Tensor, conditional : Tensor) -> Tensor:
        return self.model(x, timesteps, conditional)
    
    def on_train_batch_start(self, batch, batch_idx):
        # pytorch lightning logic for restarting epoch
        if self.stop_epoch:
            self.stop_epoch = False
            return -1
        
    def _common_step(self, batch : tuple[Tensor, Tensor, tuple[str], Tensor]) -> Tensor:        
        x0, x1, direction, is_from_cache = batch
        assert (is_from_cache == is_from_cache[0]).all(), "All tensors in the batch must have the same is_from_cache value."
        if not is_from_cache[0]:
            x0, x1 = self.encode_batch(x0, x1)
        xt, timesteps, conditional, flow = self.scheduler.sample_training_batch(x0, x1, direction)
        model_output = self(xt, timesteps, conditional)
        loss = mse_loss(model_output, flow)
        return loss
        
    def training_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        loss = self._common_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        with self.fix_validation_seed():
            with self.ema.average_parameters():
                loss = self._common_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, x_start : Tensor, direction : DIRECTIONS, scheduler_type : SCHEDULER_TYPES, num_steps : int, return_trajectory : bool, verbose : bool = True) -> Tensor:
        self.model.eval()
        
        batch_size = x_start.shape[0]
        device = x_start.device
        c = self.scheduler.get_conditional(direction, device, batch_size)
        timeschedule = self.scheduler.get_timeschedule(num_steps, scheduler_type, direction)
        x = x_start
        trajectory = [x]
        for tk_plus_one, tk in tqdm(timeschedule, desc="Sampling...", leave=False, disable=not verbose):
            t = torch.full((batch_size,), tk_plus_one if direction == 'backward' else tk, device=device)
            with self.ema.average_parameters():
                flow = self(x, t, c)
            x = self.scheduler.step(x, flow, tk_plus_one, tk, direction)
            trajectory.append(x)
            
        self.model.train()
            
        if return_trajectory:
            return torch.stack(trajectory, dim=1)
        
        return x
            
    def configure_optimizers(self):
        assert self.partial_optimizer is not None, "Optimizer must be provided during training."
        assert self.partial_lr_scheduler is not None, "Learning rate scheduler must be provided during training."
        
        optim = self.partial_optimizer(self.model.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }