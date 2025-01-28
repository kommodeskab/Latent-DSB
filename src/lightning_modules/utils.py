import torch
from torch import Tensor
import random
import math
from typing import Callable
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from src.lightning_modules.vae import VAE
from src.utils import get_ckpt_path
from src.networks import VQ

def get_symmetric_schedule(min_value : float, max_value : float, num_steps : int) -> Tensor:
    gammas = torch.zeros(num_steps)
    first_half_length = math.ceil(num_steps / 2)
    gammas[:first_half_length] = torch.linspace(min_value, max_value, first_half_length)
    gammas[-first_half_length:] = torch.flip(gammas[:first_half_length], [0])
    gammas = torch.cat([torch.tensor([0]), gammas], 0)
    return gammas

class GammaScheduler:
    def __init__(
        self,
        min_gamma : float,
        max_gamma : float,
        num_steps : int,
        T : float | None = None,
    ):
        gammas = get_symmetric_schedule(min_gamma, max_gamma, num_steps)
        if T is not None:
            gammas = T * gammas / gammas.sum()
            
        self.gammas = gammas
        self.gammas_bar = torch.cumsum(gammas, 0)
        self.final_gamma_bar = self.gammas_bar[-1]
        sigma_backward = 2 * self.gammas[1:] * self.gammas_bar[:-1] / self.gammas_bar[1:]
        sigma_forward = 2 * self.gammas[1:] * (self.final_gamma_bar - self.gammas_bar[1:]) / (self.final_gamma_bar - self.gammas_bar[:-1])
        self.sigma_backward = torch.cat([torch.tensor([0]), sigma_backward], 0)
        self.sigma_forward = torch.cat([sigma_forward, torch.tensor([0])], 0)
    
    def _get_shape_for_constant(self, x : Tensor) -> list[int]:
        return [-1] + [1] * (x.dim() - 1)

class Cache:
    def __init__(self, max_size : int):
        self.cache : list[Tensor] = []
        self.max_size = max_size
        self.batch_size = None
        
    def add(self, sample: Tensor) -> None:
        """
        Add a sample to the cache.
        """
        if self.batch_size is None:
            self.batch_size = sample.size(1)
        
        if self.is_full():
            del self.cache[0]
            
        self.cache.append(sample)
        
    def sample(self) -> Tensor:
        """
        Randomly sample a batch from the cache. The batch is a random collection of samples from the saved batches.
        """
        # each batch have shape (num_steps, batch_size, ...)
        batch = []
        for _ in range(self.batch_size):
            rand_batch = random.choice(self.cache)
            rand_sample_idx = random.randint(0, rand_batch.size(1) - 1)
            rand_sample = rand_batch[:, rand_sample_idx]
            batch.append(rand_sample)
            
        return torch.stack(batch, dim=1)
    
    def clear(self) -> None:
        """
        Clears the cache
        """
        self.cache = []
    
    def is_full(self) -> bool:
        """
        Returns whether the cache is full
        """
        return len(self) == self.max_size
        
    def __len__(self) -> int:
        return len(self.cache)