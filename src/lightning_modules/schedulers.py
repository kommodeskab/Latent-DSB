import torch
import math
from torch import Tensor, IntTensor
from typing import Tuple, Literal

def get_symmetric_schedule(min_value : float, max_value : float, num_steps : int) -> Tensor:
    gammas = torch.zeros(num_steps)
    first_half_length = math.ceil(num_steps / 2)
    gammas[:first_half_length] = torch.linspace(min_value, max_value, first_half_length)
    gammas[-first_half_length:] = torch.flip(gammas[:first_half_length], [0])
    gammas = torch.cat([torch.tensor([0]), gammas], 0)
    return gammas
    
NOISE_TYPES = Literal['training', 'inference', 'none', 'training-inference']
TARGETS = Literal['terminal', 'flow', 'semi-flow']
    
class DSBScheduler:
    def __init__(
        self,
        num_timesteps : int = 100,
        gamma_min : float | None = None,
        gamma_max : float | None = None,
        target : TARGETS = 'flow',
    ):
        assert target in ['terminal', 'flow', 'semi-flow'], "Target should be either 'terminal', 'flow' or 'semi-flow'"
        assert num_timesteps > 0, "Number of timesteps must be positive"
        self.target = target
        self.set_timesteps(num_timesteps, gamma_min, gamma_max)
        self.original_gammas_bar = self.gammas_bar.clone()
        
    @staticmethod
    def to_dim(x : Tensor, dim : int) -> Tensor:
        while x.dim() < dim:
            x = x.unsqueeze(-1)
        return x
    
    def sample_timesteps(self, batch_size : int) -> Tensor:
        random_indices = torch.randint(0, len(self.timesteps), (batch_size,))
        return self.timesteps[random_indices]
    
    def set_timesteps(self, num_timesteps : int, gamma_min : float | None = None, gamma_max : float | None = None) -> None:
        # (gamma_min + gamma_max) / 2 * num_timesteps = T
        # => gamma_min + gamma_max = 2 * T / num_timesteps
        # => gamma_min = 2 * T / num_timesteps - gamma_max
        
        if gamma_min is None and gamma_max is None:
            gamma_min = gamma_max = 1 / num_timesteps
        
        gammas = get_symmetric_schedule(gamma_min, gamma_max, num_timesteps)
        gammas_bar = torch.cumsum(gammas, 0)
        
        sampling_var = 2 * gammas[1:] * gammas_bar[:-1] / gammas_bar[1:]
        sampling_var = torch.cat([torch.tensor([0.]), sampling_var], 0)
        var = 2 * gammas
        
        if not hasattr(self, 'timesteps'):
            timesteps = torch.arange(1, num_timesteps + 1)
        else:
            old_gammas_bar = self.original_gammas_bar[1:]
            new_gammas_bar = gammas_bar[1:]
            distance_matrix = (old_gammas_bar.unsqueeze(0) - new_gammas_bar.unsqueeze(1)).abs()
            timesteps = distance_matrix.argmin(dim=1) + 1
        
        self.timesteps = timesteps
        self.gammas = gammas
        self.gammas_bar = gammas_bar
        self.sampling_var = sampling_var
        self.var = var
        
    def deterministic_sample(self, x_start : Tensor, x_end : Tensor, return_trajectory : bool = False, noise : NOISE_TYPES = 'training', noise_factor : float = 1.0) -> Tensor:
        xk = x_start
        trajectory = [xk]
        for k, _ in reversed(list(enumerate(self.timesteps, start=1))):
            if self.target == 'terminal':
                pseudo_model_output = x_end
                
            elif self.target == 'flow':
                gammas_bar = self.gammas_bar[k]
                pseudo_model_output = (xk - x_end) / gammas_bar
                
            elif self.target == 'semi-flow':
                pseudo_model_output = xk - x_end
                
            xk = self.step(xk, k, pseudo_model_output, noise, noise_factor)
            trajectory.append(xk)
        
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory if return_trajectory else trajectory[-1]
    
    def sample_batch(self, batch : Tensor, timesteps : Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        # batch.shape = (num_steps, batch_size, ...)
        batch_size = batch.size(1)
        device = batch.device
        terminal_points = batch[0]
        if timesteps is None:
            timesteps = self.sample_timesteps(batch_size).to(device)
            
        all_batches = torch.arange(batch_size).to(device)
        xt = batch[timesteps, all_batches]
        
        if self.target == "flow":
            gammas_bar = self.to_dim(self.gammas_bar, xt.dim()).to(device = xt.device, dtype=xt.dtype)
            target = (xt - terminal_points) / gammas_bar[timesteps]
            
        elif self.target == "terminal":
            target = terminal_points
            
        elif self.target == "semi-flow":
            target = xt - terminal_points
            
        return xt, timesteps, target
    
    def sample_init_batch(self, x0 : Tensor, x1 : Tensor) -> Tuple[Tensor, IntTensor, Tensor]:
        # used for the DSB-pretraining
        # we interpolate between x0 and x1 using the DSB scheduler
        # i.e. the flow is calculated as a vector poiting from xk to x1, starting from xk = x0
        trajectory = self.deterministic_sample(x0, x1, return_trajectory=True, noise='training')
        xk, timesteps, target = self.sample_batch(trajectory)
        
        return xk, timesteps, target
    
    def step(self, xk_plus_1 : Tensor, k_plus_one : int, model_output : Tensor, noise : NOISE_TYPES, noise_factor : float = 1) -> Tensor:        
        dim = xk_plus_1.dim()
        device = xk_plus_1.device
        gammas = self.to_dim(self.gammas, dim).to(device=device, dtype=xk_plus_1.dtype)
        gammas_bar = self.to_dim(self.gammas_bar, dim).to(device=device, dtype=xk_plus_1.dtype)
        
        if noise == 'training':
            std = self.var[k_plus_one] ** 0.5
        elif noise == 'inference':
            std = self.sampling_var[k_plus_one] ** 0.5
        elif noise == 'none':
            std = 0
        elif noise == 'training-inference':
            std = 0 if k_plus_one == 1 else self.var[k_plus_one] ** 0.5
                
        step_size = gammas[k_plus_one]
        
        if self.target == "flow":
            direction = model_output
        elif self.target == "terminal":
            direction = (xk_plus_1 - model_output) / gammas_bar[k_plus_one]
        elif self.target == "semi-flow":
            direction = model_output / gammas_bar[k_plus_one]
        
        mu = xk_plus_1 - step_size * direction
        noise = torch.randn_like(xk_plus_1) * std * noise_factor
        return mu + noise

if __name__ == "__main__":
    pass