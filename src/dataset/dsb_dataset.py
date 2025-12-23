from torch.utils.data import IterableDataset, Dataset
import random
import gc
from torch import Tensor
import multiprocessing as mp

class Cache:
    def __init__(self, maxlen: int):
        assert maxlen >= 0, "maxlen must be non-negative"
        self.maxlen = maxlen
        self.manager = mp.Manager()
        self.cache = self.manager.list()  # shared list across processes
    
    def add(self, x0: Tensor, x1: Tensor) -> None:
        x0_s = x0.detach().cpu().clone()
        x1_s = x1.detach().cpu().clone()
        
        # enforce maxlen manually for manager.list
        if self.maxlen > 0 and len(self.cache) >= self.maxlen:
            self.cache.pop(0)
            
        self.cache.append((x0_s, x1_s))
        gc.collect()
        
    def __len__(self) -> int:
        return len(self.cache)

    def is_full(self) -> bool:
        if self.maxlen == 0:
            return False
        
        return len(self.cache) >= self.maxlen
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.cache[idx]

class DSBDataset(IterableDataset):
    def __init__(
        self,
        x0_dataset: Dataset,
        x1_dataset: Dataset,
        cache_size: int,
    ):
        super().__init__()
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        self.forward_cache = Cache(maxlen = cache_size // 2)
        self.backward_cache = Cache(maxlen = cache_size // 2)
        
    @staticmethod
    def random_sample(dataset: Cache | Dataset) -> tuple[Tensor, Tensor] | Tensor:
        idx = random.randint(0, len(dataset) - 1)
        return dataset[idx]
    
    @property
    def caches_are_full(self) -> bool:
        return self.forward_cache.is_full() and self.backward_cache.is_full()

    def __iter__(self):
        while True:
            if self.caches_are_full:
                # sample pairs from caches
                x0_f, x1_f = self.random_sample(self.forward_cache)
                x0_b, x1_b = self.random_sample(self.backward_cache)
                yield x0_f, x1_f, x0_b, x1_b
            else:
                # randomly sample pairs from datasets
                x1_f, x1_b = self.random_sample(self.x1_dataset), self.random_sample(self.x1_dataset)
                x0_f, x0_b = self.random_sample(self.x0_dataset), self.random_sample(self.x0_dataset)
                yield x0_f, x1_f, x0_b, x1_b