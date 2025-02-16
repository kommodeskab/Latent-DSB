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
    def __init__(self, num_timesteps : int, gamma_frac : float = 0.001):
        self.num_train_timesteps = num_timesteps
        self.set_timesteps(num_timesteps, gamma_frac)
        self.original_gammas_bar = self.gammas_bar.clone()
    
    def set_timesteps(self, num_timesteps : int, gamma_frac : float = 1.0) -> None:
        gammas = get_symmetric_schedule(gamma_frac, 1, num_timesteps)
        gammas = gammas / gammas.sum()
        gammas_bar = torch.cumsum(gammas, 0)
        var = 2 * gammas[1:] * gammas_bar[:-1] / gammas_bar[1:]
        var = torch.cat([torch.tensor([0]), var, torch.tensor([0])], 0)
        if not hasattr(self, 'original_gammas_bar'):
            timesteps = torch.arange(1, num_timesteps + 1)
        else:
            # for each gamma, find the index of the original gamma that is closest to it
            timesteps = torch.tensor([torch.argmin(torch.abs(self.original_gammas_bar[1:] - gamma_bar)) for gamma_bar in gammas_bar[1:]]) + 1
            timesteps = timesteps.clamp(min = 1)
        
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
    def __init__(self, num_timesteps : int = 1000, gamma_frac : float = 0.001, ode : bool = True):
        super().__init__(num_timesteps, gamma_frac)
        self.ode = ode
    
    def sample_batch(self, batch : Tensor) -> Tuple[Tensor, IntTensor, Tensor]:
        batch_size = batch.size(0)
        device = batch.device
        timesteps = self.sample_timesteps(batch_size).to(device)
        gammas_bar = self.gammas_bar.to(device)
        gammas_bar = self.to_dim(gammas_bar, batch.dim())
        gammas_bar = gammas_bar[timesteps]
        noise = torch.randn_like(batch)
        target = noise - batch
        xt = (1 - gammas_bar) * batch + gammas_bar * noise
        var = self.to_dim(self.var, batch.dim()).to(device)
        # var = [0, var[0 | 1], var[1 | 2], ...]
        std = 0 if self.ode else var[timesteps + 1] ** 0.5
        xt = xt + std * torch.randn_like(xt)
        return xt, timesteps, target
    
    def step(self, xt_plus_1 : Tensor, t_plus_1 : int, model_output : Tensor) -> Tensor:
        """
        Predict x_t | x_{t + 1}
        """
        t_plus_1 = torch.where(self.timesteps == t_plus_1)[0].min() + 1
        gammas_bar = self.gammas_bar.to(xt_plus_1.device)
        delta_t = gammas_bar[t_plus_1 - 1] - gammas_bar[t_plus_1]
        mu = xt_plus_1 + delta_t * model_output
        # var = [0, var[0 | 1], var[1 | 2], ...]
        std = 0 if self.ode else self.var[t_plus_1] ** 0.5
        return mu + std * torch.randn_like(mu)
    
class DSBScheduler(BaseScheduler):
    def __init__(
        self,
        num_steps : int,
        gamma_frac : float = 1.0,
        target : Literal['terminal', 'flow'] = 'terminal',
    ):
        super().__init__(num_steps, gamma_frac)
        self.target = target
    
    def sample_batch(self, batch : Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # batch.shape = (num_steps, batch_size, ...)
        batch_size = batch.size(1)
        device = batch.device
        terminal_points = batch[0]
        timesteps = self.sample_timesteps(batch_size).to(device)
        all_batches = torch.arange(batch_size).to(device)
        xt = batch[timesteps, all_batches]
        
        if self.target == "terminal":
            target = terminal_points
        else:
            gammas_bar = self.to_dim(self.gammas_bar, xt.dim()).to(device)
            target = (terminal_points - xt) / gammas_bar[timesteps]
            
        return xt, timesteps, target
    
    def step(self, xk_plus_1 : Tensor, k_plus_one : int, model_output : Tensor):
        dim = xk_plus_1.dim()
        device = xk_plus_1.device
        gammas = self.to_dim(self.gammas, dim).to(device)
        gammas_bar = self.to_dim(self.gammas_bar, dim).to(device)
        
        std = self.var[k_plus_one] ** 0.5
        step_size = gammas[k_plus_one]
        
        if self.target == "terminal":
            direction = (model_output - xk_plus_1) / gammas_bar[k_plus_one]
        else:
            direction = model_output
        
        mu = xk_plus_1 + step_size * direction
        noise = torch.randn_like(xk_plus_1) * std
        return mu + noise
    
if __name__ == "__main__":
    scheduler = FMScheduler(num_timesteps=20)
    print(scheduler.var)
    print(scheduler.timesteps)
