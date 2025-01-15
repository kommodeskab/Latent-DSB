import torch
from torch import Tensor
import random
import math
from typing import Callable
from torch import device


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

class VPScheduler:
    def __init__(
        self,
        min_value : float,
        max_value : float,
        num_steps : int,
    ):
        gammas = get_symmetric_schedule(min_value, max_value, num_steps)
        sigmas2 = torch.cumsum(gammas, 0)
        alphas = torch.sqrt(1 - sigmas2)
        alphas_ts = alphas[1:] / alphas[:-1]
        sigmas2_ts = sigmas2[1:] - alphas_ts ** 2 * sigmas2[:-1]
        sigmas2_Q = sigmas2_ts * sigmas2[1:] / sigmas2[:-1]

class Cache:
    def __init__(self, max_size : int):
        self.cache = []
        self.max_size = max_size
        
    def add(self, sample: Tensor) -> None:
        """
        Add a sample to the cache.
        """
        if len(self) >= self.max_size:
            del self.cache[0]
            
        self.cache.append(sample)
        
    def sample(self) -> Tensor:
        """
        Randomly sample a sample from the cache. The sample is removed from the cache.
        """
        randint = random.randint(0, len(self.cache) - 1)
        return self.cache[randint]
    
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

_NETWORK = Callable[[Tensor], Tensor]

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

class StableDiffusionVAE(torch.nn.Module):
    def __init__(
        self,
        version : str
        ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(f"stable-diffusion-vae/{version}")
        self.vae.to(device)
        num_params = sum(p.numel() for p in self.vae.parameters())
        print(f"Loaded stable-diffusion-vae/{version} with {num_params} parameters")
        
    @torch.no_grad()
    def encode(self, x : Tensor) -> Tensor:
        dist : DiagonalGaussianDistribution = self.vae.encode(x).latent_dist
        return dist.mean.detach()
    
    @torch.no_grad()
    def decode(self, z : Tensor) -> Tensor:
        return self.vae.decode(z).sample.detach().clip(-1, 1)
    
class StableDiffusionEncoder:
    def __init__(self, vae : StableDiffusionVAE):
        self.vae = vae
        
    def __call__(self, x : Tensor) -> Tensor:
        return self.vae.encode(x)
    
class StableDiffusionDecoder:
    def __init__(self, vae : StableDiffusionVAE):
        self.vae = vae
        
    def __call__(self, z : Tensor) -> Tensor:
        return self.vae.decode(z)

def get_encoder_decoder(id : str) -> tuple[_NETWORK, _NETWORK]:
    match id:
        case "downsampler":
            encoder = torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
            decoder = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        case "identity":
            encoder = torch.nn.Identity()
            decoder = torch.nn.Identity()
            
        case "stable-diffusion-1.4":
            vae = StableDiffusionVAE("1.4")
            encoder = StableDiffusionEncoder(vae)
            decoder = StableDiffusionDecoder(vae)
            
        case "stable-diffusion-3.5":
            vae = StableDiffusionVAE("3.5")
            encoder = StableDiffusionEncoder(vae)
            decoder = StableDiffusionDecoder(vae)
            
        case _:
            raise ValueError(f"Unknown id {id}")
            
    return encoder, decoder