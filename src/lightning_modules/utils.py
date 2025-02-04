import torch
from torch import Tensor
import random
import math
import os
from torch.utils.data import DataLoader, Dataset
import hashlib
from src.networks.encoders import BaseEncoderDecoder
from tqdm import tqdm

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
        # batch.shape (num_steps, batch_size, ...)
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
    
    
def hash_obj(obj):
    return hashlib.md5(str(obj).encode()).hexdigest()

def calculate_latent_mean_std(
    encoder_decoder : BaseEncoderDecoder,
    datasets: list[Dataset],
    max_n_batches: int = 50,
    ):
    """
    Given an encoder/decoder and a list of datasets, calculate the mean and standard deviation of the latent space. 
    If multiple datatets are provided, the mean and standard deviation are calculated across all datasets and weighted by the number of samples in each dataset.
    """
    root_dir = 'latent_means'
    os.makedirs(root_dir, exist_ok=True)
    device = next(encoder_decoder.parameters()).device
    
    dataset_sizes_sum = sum([len(dataset) for dataset in datasets])
    
    latent_mean = []
    latent_std = []
    
    for dataset in datasets:
        dataset_hash = hash_obj(encoder_decoder) + hash_obj(dataset)
        file_name = f'{root_dir}/latent_mean_std_{dataset_hash}.pt'
        if os.path.exists(file_name):
            file = torch.load(file_name)
            print(f'Loaded latent mean and std from {file_name}')
            dataset_latent_mean = file['latent_mean']
            dataset_latent_std = file['latent_std']
        else:
            dataset_latent_mean = []
            dataset_latent_std = []
            dataloader = DataLoader(dataset, batch_size = 128)
            for i, batch in tqdm(enumerate(dataloader), total=max_n_batches, desc=f'Calculating latent mean and std for dataset {dataset_hash}'):
                if i > max_n_batches:
                    break
                with torch.no_grad():
                    latent = encoder_decoder.encode(batch.to(device))
                dataset_latent_mean.append(latent.mean(dim=0))
                dataset_latent_std.append(latent.std(dim=0))
                
            dataset_latent_mean = torch.stack(dataset_latent_mean).mean(dim=0)
            dataset_latent_std = torch.stack(dataset_latent_std).mean(dim=0)
            torch.save({'latent_mean': dataset_latent_mean, 'latent_std': dataset_latent_std}, file_name)
            print(f'Saved latent mean and std to {file_name}')
            
        latent_mean.append(len(dataset) / dataset_sizes_sum * dataset_latent_mean)
        latent_std.append(len(dataset) / dataset_sizes_sum * dataset_latent_std)
        
    latent_mean = torch.stack(latent_mean).sum(dim=0)
    latent_std = torch.stack(latent_std).sum(dim=0)
    
    return latent_mean, latent_std