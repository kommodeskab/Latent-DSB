from torch import Tensor
import torch
from typing import Literal, get_args
import math
import numpy as np

DIRECTIONS = Literal['forward', 'backward']
SCHEDULER_TYPES = Literal['linear', 'cosine']

class DSBScheduler:
    def __init__(
        self, 
        epsilon : float = 2.0,
        condition_on_start : bool = False,
        **kwargs
        ):
        self.epsilon = epsilon
        self.condition_on_start = condition_on_start
    
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
        xt_var = self.epsilon * t * (1 - t)
        noise = torch.randn_like(xt_mean)
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
        
        if self.condition_on_start:
            # concatenate x0/x1 to xt along the channel dimension
            start = torch.zeros_like(xt)
            start[mask] = x0[mask]
            start[~mask] = x1[~mask]
            xt = torch.cat([xt, start], dim=1)
        
        return xt, timesteps, conditional, flow
        
    def get_dummy_trajectory(self, x0 : Tensor, x1 : Tensor, trajectory_length : int) -> Tensor:
        """Generate a dummy trajectory by sampling from the initial and final states.

        Args:
            x0 (Tensor): The initial state tensor (sampled from p_0 or p_{data}).
            x1 (Tensor): The final state tensor (sampled from p_1 or p_{prior}).
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
            xnext_var = self.epsilon * delta_t * tk / tk_plus_one
        elif direction == 'forward':
            xnext_var = self.epsilon * delta_t * (1 - tk_plus_one) / (1 - tk)
        
        noise = torch.randn_like(xnext_mean)

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