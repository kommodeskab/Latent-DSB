from torch import Tensor
import torch
from typing import Literal, get_args
import math
import numpy as np
from typing import Dict, Optional

DIRECTIONS = Literal['forward', 'backward']
SCHEDULER_TYPES = Literal['linear', 'cosine']
TensorDict = Dict[str, Tensor]

class BaseScheduler:
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

class DSBScheduler(BaseScheduler):
    def __init__(
        self, 
        epsilon : float,
        target: Literal['drift', 'scaled_drift', 'terminal'] = 'drift',
        condition_on_start: bool = False,
        **kwargs
        ):
        super().__init__()
        self.epsilon = epsilon
        self.target = target
        self.condition_on_start = condition_on_start
        
    def get_conditional(self, direction: DIRECTIONS, batch_size: int, device: torch.device) -> Tensor:
        if direction == 'forward':
            return torch.ones((batch_size,), dtype=torch.long, device=device)
        elif direction == 'backward':
            return torch.zeros((batch_size,), dtype=torch.long, device=device)
        
        return direction
    
    def sample_training_batch(self, x0 : Tensor, x1 : Tensor, direction : DIRECTIONS, timesteps : Optional[Tensor] = None) -> TensorDict:
        assert direction in get_args(DIRECTIONS), f"Unsupported direction. Use either: {get_args(DIRECTIONS)}"
        
        batch_size = x0.size(0)
        
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=x0.device)
            
        if direction == 'forward':
            timesteps = timesteps.clamp(max=0.99)
        else:
            timesteps = timesteps.clamp(min=0.01)
            
        t = self.to_dim(timesteps, x0.dim())
        mu_t = (1 - t) * x0 + t * x1
        sigma_t = self.epsilon * t * (1 - t)
        noise = torch.randn_like(mu_t)
        xt = mu_t + sigma_t ** 0.5 * noise
        
        accumulated_time = (1 - t) if direction == 'forward' else t
        terminal = x1 if direction == 'forward' else x0
        
        if self.target == 'terminal':
            target = terminal
        elif self.target == 'drift':
            target = (terminal - xt) / accumulated_time
        elif self.target == 'scaled_drift':
            target = (terminal - xt) / accumulated_time.sqrt()
        else:
            raise ValueError(f"Unsupported target type. Use either: {get_args(Literal['drift', 'scaled_drift', 'terminal'])}")
        
        conditional = self.get_conditional(direction, batch_size, x0.device)
        
        if self.condition_on_start:
            start = x0 if direction == 'forward' else x1
            xt = torch.cat([xt, start], dim=1)
            
        return {
            "xt": xt,
            "target": target,
            "timesteps": timesteps,
            "conditional": conditional,
        }
    
    def step(self, xt : Tensor, model_output : Tensor, tk_plus_one : float, tk : float, direction : DIRECTIONS) -> Tensor:
        assert direction in get_args(DIRECTIONS), f"Unsupported direction. Use either: {get_args(DIRECTIONS)}"
        assert tk_plus_one > tk, f"tk_plus_one ({tk_plus_one}) must be greater than tk ({tk})."
        
        delta_t = tk_plus_one - tk # the step size
        accumulated_time = (1 - tk) if direction == 'forward' else tk_plus_one
        
        if self.target == 'terminal':
            drift = (model_output - xt) / accumulated_time
        elif self.target == 'drift':
            drift = model_output
        elif self.target == 'scaled_drift':
            drift = model_output / accumulated_time ** 0.5
        else:
            raise ValueError(f"Unsupported target type. Use either: {get_args(Literal['drift', 'scaled_drift', 'terminal'])}")
        
        if direction == 'backward':
            xnext_mean = xt + delta_t * drift
            xnext_var = self.epsilon * delta_t * tk / tk_plus_one
        elif direction == 'forward':
            xnext_mean = xt + delta_t * drift
            xnext_var = self.epsilon * delta_t * (1 - tk_plus_one) / (1 - tk)
        
        noise = torch.randn_like(xnext_mean)

        xnext = xnext_mean + xnext_var ** 0.5 * noise
        
        return xnext

class BaggeScheduler(BaseScheduler):
    def __init__(
        self,
        epsilon : float,
    ):
        super().__init__()
        self.epsilon = epsilon
        
    def sample_training_batch(self, x0: Tensor, x1: Tensor, timesteps: Optional[Tensor] = None) -> TensorDict:   
        batch_size = x0.size(0)
        
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=x0.device)
             
        t = self.to_dim(timesteps, x0.dim())
        mu_t = (1 - t) * x0 + t * x1
        sigma_t = self.epsilon * t * (1 - t)
        noise = torch.randn_like(mu_t)
        xt = mu_t + sigma_t ** 0.5 * noise
        flow = x1 - x0
        
        return {
            "xt": xt,
            "noise": noise,
            "flow": flow,
            "timesteps": timesteps,
        }
        
    def step(self, xt: Tensor, flow: Tensor, noise: Tensor, tk_plus_one: float, tk: float, direction : DIRECTIONS) -> Tensor:
        delta_t = tk_plus_one - tk
        eps = self.epsilon
        
        if direction == 'forward':
            drift = flow - (eps * tk / (1 - tk)) ** 0.5 * noise
            mu_next = xt + delta_t * drift
            sigma_next = eps * delta_t * (1 - tk_plus_one) / (1 - tk)
        elif direction == 'backward':
            drift = flow + (eps * (1 - tk_plus_one) / tk_plus_one) ** 0.5 * noise
            mu_next = xt - delta_t * drift
            sigma_next = eps * delta_t * tk / tk_plus_one
            
        xt_next = mu_next + sigma_next ** 0.5 * torch.randn_like(xt)
        return xt_next