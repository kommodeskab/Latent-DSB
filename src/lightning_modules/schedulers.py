import torch
import math
from torch import Tensor, IntTensor
from typing import Tuple, Literal, Iterable

def get_symmetric_schedule(min_value : float, max_value : float, num_steps : int) -> Tensor:
    gammas = torch.zeros(num_steps)
    first_half_length = math.ceil(num_steps / 2)
    gammas[:first_half_length] = torch.linspace(min_value, max_value, first_half_length)
    gammas[-first_half_length:] = torch.flip(gammas[:first_half_length], [0])
    gammas = torch.cat([torch.tensor([0]), gammas], 0)
    return gammas

class BaseScheduler:
    def __init__(self, num_timesteps : int, gamma_min : float | None = None, gamma_max : float | None = None):
        assert num_timesteps > 0, "Number of timesteps must be positive"
        self.num_train_timesteps = num_timesteps
        self.set_timesteps(num_timesteps, gamma_min, gamma_max)
    
    def set_timesteps(self, num_timesteps : int, gamma_min : float | None = None, gamma_max : float | None = None) -> None:
        if gamma_min is None and gamma_max is None:
            gamma_min = gamma_max = 1 / num_timesteps
        
        gammas = get_symmetric_schedule(gamma_min, gamma_max, num_timesteps)
        gammas_bar = torch.cumsum(gammas, 0)
        
        sampling_var = 2 * gammas[1:] * gammas_bar[:-1] / gammas_bar[1:]
        sampling_var = torch.cat([torch.tensor([0]), sampling_var], 0)
        var = 2 * gammas
        
        timesteps = torch.arange(1, num_timesteps + 1)
        
        self.timesteps = timesteps
        self.gammas = gammas
        self.gammas_bar = gammas_bar
        self.sampling_var = sampling_var
        self.var = var

    def sample_timesteps(self, batch_size : int) -> Tensor:
        random_indices = torch.randint(0, len(self.timesteps), (batch_size,))
        return self.timesteps[random_indices]
    
    def sample_batch(self, *args) -> Tuple[Tensor, IntTensor, Tensor]:
        raise NotImplementedError
    
    def step(self, xt_plus_1 : Tensor, t_plus_1 : int, model_output : Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def to_dim(x : Tensor, dim : int) -> Tensor:
        while x.dim() < dim:
            x = x.unsqueeze(-1)
        return x
    
NOISE_TYPES = Literal['training', 'inference', 'none', 'training-inference']
TARGETS = Literal['terminal', 'flow', 'semi-flow']
    
class DSBScheduler(BaseScheduler):
    def __init__(
        self,
        num_timesteps : int = 100,
        gamma_min : float | None = None,
        gamma_max : float | None = None,
        target : TARGETS = 'flow',
    ):
        assert target in ['terminal', 'flow', 'semi-flow'], "Target should be either 'terminal', 'flow' or 'semi-flow'"
        super().__init__(num_timesteps, gamma_min, gamma_max)
        self.target = target
        
    def deterministic_sample(self, x_start : Tensor, x_end : Tensor, return_trajectory : bool = False, noise : NOISE_TYPES = 'training') -> Tensor:
        xk = x_start
        trajectory = [xk]
        for k in reversed(self.timesteps):
            if self.target == 'terminal':
                pseudo_model_output = x_end
                
            elif self.target == 'flow':
                gammas_bar = self.gammas_bar[k]
                pseudo_model_output = (xk - x_end) / gammas_bar
                
            elif self.target == 'semi-flow':
                pseudo_model_output = xk - x_end
                
            xk = self.step(xk, k, pseudo_model_output, noise)
            trajectory.append(xk)
        
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory if return_trajectory else trajectory[-1]
        
    def forward_process(self, x0 : Tensor, x1 : Tensor, timesteps : IntTensor | int) -> Tuple[Tensor, Tensor]:
        trajectory = self.deterministic_sample(x0, x1, return_trajectory=True, noise='training')
        all_batches = torch.arange(trajectory.size(1)).to(trajectory.device)
        xt = trajectory[timesteps, all_batches]

        return xt, ...
    
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
            gammas_bar = self.to_dim(self.gammas_bar, xt.dim()).to(device)
            target = (xt - terminal_points) / gammas_bar[timesteps]
            
        elif self.target == "terminal":
            target = terminal_points
            
        elif self.target == "semi-flow":
            target = xt - terminal_points
            
        return xt, timesteps, target
    
    def step(self, xk_plus_1 : Tensor, k_plus_one : int, model_output : Tensor, noise : NOISE_TYPES) -> Tensor:
        dim = xk_plus_1.dim()
        device = xk_plus_1.device
        gammas = self.to_dim(self.gammas, dim).to(device)
        gammas_bar = self.to_dim(self.gammas_bar, dim).to(device)
        
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
        noise = torch.randn_like(xk_plus_1) * std
        return mu + noise

if __name__ == "__main__":
    pass