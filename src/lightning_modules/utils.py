import torch
from torch import Tensor
from typing import Any
from collections import deque

class DSBCache:
    def __init__(self, max_size : int, batch_size : int):
        self.cache = deque(maxlen=max_size)
        self.batch_size = batch_size
        
    def add(self, sample : Tensor) -> None:
        # sample.shape (num_steps, batch_size, ...)
        batch_size = sample.size(1)
        for i in range(batch_size):
            self.cache.append(sample[:, i])
            
    def sample(self) -> Tensor:
        rand_idxs = torch.randint(0, len(self.cache), (self.batch_size,))
        batch = [self.cache[idx] for idx in rand_idxs]
        batch = torch.stack(batch, dim=1)
        return batch
    
    def clear(self) -> None:
        self.cache.clear()
        
    def is_full(self) -> bool:
        return len(self.cache) >= self.cache.maxlen
        
    def __len__(self) -> int:
        return len(self.cache)
    
def sort_dict_by_model(state_dict : dict[str, Any], models_to_keep : list[str]):
    def is_ok(key : str) -> bool:
        return any([key.startswith(m) for m in models_to_keep])
    new_state_dict = {k : v for k, v in state_dict.items() if is_ok(k)}
    return new_state_dict