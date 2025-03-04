import torch
from torch import Tensor
import random
from typing import Any

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
            
        for i in range(self.batch_size):
            if self.is_full():
                del self.cache[self.rand_idx()]
            
            self.cache.append(sample[:, i])
        
    def sample(self) -> Tensor:
        """
        Randomly sample a batch from the cache. The batch is a random collection of samples from the saved batches.
        """
        # batch.shape (num_steps, batch_size, ...)
        batch = []
        for _ in range(self.batch_size):
            idx = self.rand_idx()
            batch.append(self.cache[idx])
            
        return torch.stack(batch, dim=1)
    
    def clear(self) -> None:
        self.cache = []
        
    def rand_idx(self) -> int:
        return random.randint(0, len(self.cache) - 1)
    
    def is_full(self) -> bool:
        return len(self) == self.max_size
        
    def __len__(self) -> int:
        return len(self.cache)
    
def sort_dict_by_model(state_dict : dict[str, Any], models_to_keep : list[str]):
    def is_ok(key : str) -> bool:
        return any([key.startswith(m) for m in models_to_keep])
    new_state_dict = {k : v for k, v in state_dict.items() if is_ok(k)}
    return new_state_dict