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

class BaseScheduler:
    def __init__(self, num_timesteps : int, gamma_frac : float = 0.001, T : float = 1.0):
        self.num_train_timesteps = num_timesteps
        self.set_timesteps(num_timesteps, gamma_frac, T)
        self.original_gammas_bar = self.gammas_bar.clone()
    
    def set_timesteps(self, num_timesteps : int, gamma_frac : float = 1.0, T : float = 1.0) -> None:
        gammas = get_symmetric_schedule(gamma_frac, 1, num_timesteps)
        gammas = T * gammas / gammas.sum()
        gammas_bar = torch.cumsum(gammas, 0)
        var = 2 * gammas[1:] * gammas_bar[:-1] / gammas_bar[1:]
        var = torch.cat([torch.tensor([0]), var], 0)
        timesteps = torch.arange(1, num_timesteps + 1)
        
        self.timesteps = timesteps
        self.gammas = gammas
        self.gammas_bar = gammas_bar
        self.var = var
    
    def sample_timesteps(self, batch_size : int) -> Tensor:
        random_indices = torch.randint(0, len(self.timesteps), (batch_size,))
        return self.timesteps[random_indices]
    
    @staticmethod
    def to_dim(x : Tensor, dim : int) -> Tensor:
        while x.dim() < dim:
            x = x.unsqueeze(-1)
        return x

class FMScheduler(BaseScheduler):
    def __init__(
        self, 
        num_timesteps : int = 1000, 
        gamma_frac : float = 1.0, 
        target : Literal['terminal', 'flow'] = 'flow', 
        T : float = 1.0
    ):
        super().__init__(num_timesteps, gamma_frac, T)
        assert target in ['terminal', 'flow'], "Target should be either 'terminal' or 'flow'"
        self.target = target
        self.gammas_bar[-1] = 1 - 1e-4
        
    def forward_process(self, x0 : Tensor, x1 : Tensor, timesteps : IntTensor | int) -> Tuple[Tensor, Tensor]:
        device = x0.device
        dim = x0.dim()
        
        gammas_bar = self.gammas_bar.to(device)
        gammas_bar = self.to_dim(gammas_bar, dim)
        gammas_bar = gammas_bar[timesteps]
        xt = (1 - gammas_bar) * x0 + gammas_bar * x1
        
        if self.target == 'flow':
            target = x1 - x0
        else:
            target = x0
            
        return xt, target
    
    def sample_batch(self, x0 : Tensor, x1 : Tensor) -> Tuple[Tensor, IntTensor, Tensor]:
        batch_size = x0.size(0)
        device = x0.device
        timesteps = self.sample_timesteps(batch_size).to(device)
        xt, target = self.forward_process(x0, x1, timesteps)
        
        return xt, timesteps, target
    
    def step(self, xt_plus_1 : Tensor, t_plus_1 : int, model_output : Tensor) -> Tensor:
        """
        Predict x_t | x_{t + 1}
        """
        gammas_bar = self.gammas_bar.to(xt_plus_1.device)
        delta_t = gammas_bar[t_plus_1] - gammas_bar[t_plus_1 - 1]
        
        if self.target == "flow":
            direction = model_output
        else:
            direction = (xt_plus_1 - model_output) / gammas_bar[t_plus_1]
        
        xt = xt_plus_1 - delta_t * direction
        return xt
    
class DSBScheduler(BaseScheduler):
    def __init__(
        self,
        num_timesteps : int,
        gamma_frac : float = 1.0,
        target : Literal['terminal', 'flow'] = 'terminal',
        T : float = 1.0
    ):
        super().__init__(num_timesteps, gamma_frac, T)
        assert target in ['terminal', 'flow'], "Target should be either 'terminal' or 'flow'"
        self.target = target
    
    def sample_batch(self, batch : Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # batch.shape = (num_steps, batch_size, ...)
        batch_size = batch.size(1)
        device = batch.device
        terminal_points = batch[0]
        timesteps = self.sample_timesteps(batch_size).to(device)
        all_batches = torch.arange(batch_size).to(device)
        xt = batch[timesteps, all_batches]
        
        if self.target == "flow":
            gammas_bar = self.to_dim(self.gammas_bar, xt.dim()).to(device)
            target = (xt - terminal_points) / gammas_bar[timesteps]
        elif self.target == "terminal":
            target = terminal_points
            
        return xt, timesteps, target
    
    def step(self, xk_plus_1 : Tensor, k_plus_one : int, model_output : Tensor) -> Tensor:
        dim = xk_plus_1.dim()
        device = xk_plus_1.device
        gammas = self.to_dim(self.gammas, dim).to(device)
        gammas_bar = self.to_dim(self.gammas_bar, dim).to(device)
        
        std = self.var[k_plus_one] ** 0.5
        step_size = gammas[k_plus_one]
        
        if self.target == "flow":
            direction = model_output
        elif self.target == "terminal":
            direction = (xk_plus_1 - model_output) / gammas_bar[k_plus_one]
        
        mu = xk_plus_1 - step_size * direction
        noise = torch.randn_like(xk_plus_1) * std
        return mu + noise
        
class I2Scheduler(BaseScheduler):
    def __init__(
        self,
        num_timesteps : int, 
        gamma_frac : float = 1.0, 
        target : Literal['terminal', 'flow'] = 'flow',
        T : float = 1.0,    
    ):
        super().__init__(num_timesteps, gamma_frac, T)
        assert target in ['terminal', 'flow'], "Target should be either 'terminal' or 'flow'"
        self.target = target
        
    def forward_process(self, x0 : Tensor, x1 : Tensor, timesteps : IntTensor | int) -> tuple[Tensor, Tensor]:
        device = x0.device
        dim = x0.dim()
        
        gammas_bar = self.gammas_bar.to(device)
        gammas_bar = self.to_dim(gammas_bar, dim)
        sigmas = gammas_bar[timesteps]
        mu = (1 - sigmas) * x0 + sigmas * x1
        std = (sigmas * (1 - sigmas)) ** 0.5
        xt = mu + std * torch.randn_like(mu)
        
        if self.target == 'terminal':
            target = x0
        else:
            target = (xt - x0) / sigmas
            
        return xt, target
        
    def sample_batch(self, x0 : Tensor, x1 : Tensor) -> tuple[Tensor, IntTensor, Tensor]:
        batch_size = x0.size(0)
        device = x0.device
        timesteps = self.sample_timesteps(batch_size).to(device)
        xt, target = self.forward_process(x0, x1, timesteps)
        return xt, timesteps, target    
    
    def step(self, xt_plus_1 : Tensor, t_plus_1 : int, model_output : Tensor) -> Tensor:
        device = xt_plus_1.device
        
        sigmas = self.gammas_bar.to(device)
        sigma = sigmas[t_plus_1 - 1]
        alfas = self.gammas.to(device)
        alfa = alfas[t_plus_1]
        
        if self.target == 'terminal':
            x0_hat = model_output
        else:
            sigma_plus_1 = sigmas[t_plus_1]
            x0_hat = xt_plus_1 - model_output * sigma_plus_1
        
        denom = alfa + sigma
        mu = alfa / denom * x0_hat + sigma / denom * xt_plus_1
        std = (sigma * alfa / denom) ** 0.5
        return mu + std * torch.randn_like(mu)
    
class ReFlowScheduler(BaseScheduler):
    def __init__(self, num_timesteps : int, gamma_frac : float = 1.0):
        super().__init__(num_timesteps, gamma_frac)
        
    def sample_batch(self, x0 : Tensor, x1 : Tensor, training_backward : bool = True) -> tuple[Tensor, Tensor, Tensor]:
        """
        When training backward, the provided trajectory is sampled going from x0 --> x1. 
        When training forward, the provided trajectory is sampled going from x1 --> x0.
        """
        batch_size = x0.size(0)
        
        timesteps = self.sample_timesteps(batch_size)
        if not training_backward:
            timesteps = timesteps - 1
        
        dim = x0.dim()
        gammas_bar = self.gammas_bar.to(x0.device)
        gammas_bar = self.to_dim(gammas_bar, dim)
        gammas_bar = gammas_bar[timesteps]
        xt = (1 - gammas_bar) * x0 + gammas_bar * x1
        target = x1 - x0
        return xt, timesteps, target
    
    def get_timesteps_for_sampling(self, sampling_backward : bool = True) -> Tensor:
        """
        If sampling backward, the timesteps goes from T, T - 1, ..., 1.
        If sampling forward, the timesteps goes from 0, 1, ..., T - 1.
        """
        if sampling_backward:
            return reversed(self.timesteps)
        else:
            return self.timesteps - 1
        
    def step(self, xk : Tensor, k : Tensor, model_output : Tensor, sampling_backward : bool = True) -> Tensor:
        # model_out = x1 - x0
        gammas_bar = self.gammas_bar.to(xk.device)
        next_k = k - 1 if sampling_backward else k + 1
        delta_t = gammas_bar[next_k] - gammas_bar[k]
        x_next = xk + delta_t * model_output
        
        return x_next
    
    
if __name__ == "__main__":
    pass