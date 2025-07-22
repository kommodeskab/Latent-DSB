from src.lightning_modules.esdsb import DIRECTIONS
from torch.utils.data import Dataset, IterableDataset
from torch import Tensor
import torch
import random
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Generator
from src.data_modules.base_dm import split_dataset
import multiprocessing as mp
from multiprocessing.managers import ListProxy

def infinite_random_cycle(length : int) -> Generator[int, None, None]:
    while True:
        indices = torch.randperm(length).tolist()
        for i in indices:
            yield i
            
class NormalDataset(Dataset):
    def __init__(self, shape : tuple[int, ...], num_samples : int):
        self.shape = shape
        self.num_samples = num_samples
    
    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return torch.randn(self.shape, dtype=torch.float32)

class DSBDataset(IterableDataset):
    def __init__(
        self, 
        x0_dataset : Dataset, 
        x1_dataset : Dataset,
    ):
        super().__init__()
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        self.x0_cycle = infinite_random_cycle(len(x0_dataset))
        self.x1_cycle = infinite_random_cycle(len(x1_dataset))
        
        manager = mp.Manager()
        self.forward_cache : ListProxy[tuple[Tensor, Tensor]] = manager.list()
        self.backward_cache : ListProxy[tuple[Tensor, Tensor]] = manager.list()
        
    def add_to_cache(self, x0 : Tensor, x1 : Tensor, direction : DIRECTIONS) -> None:
        cache = self.forward_cache if direction == 'forward' else self.backward_cache
        batch_size = x0.shape[0]
        for i in range(batch_size):
            x0_sample, x1_sample = x0[i], x1[i]
            cache.append((x0_sample, x1_sample))
    
    def __iter__(self) -> Generator[Tensor, Tensor, str]:
        while True:
            direction = random.choice(['backward', 'forward']) # TODO: always have 'backward' and 'forward'
            
            if len(self.forward_cache) > 0: # if there is something in forward cache, there is also something in backward cache
                cache = self.forward_cache if direction == 'forward' else self.backward_cache            
                x0, x1 = random.choice(cache)
                yield x0, x1, direction
            else:
                x0 = self.x0_dataset[next(self.x0_cycle)]
                x1 = self.x1_dataset[next(self.x1_cycle)]
                yield x0, x1, direction

class EvalDataset(Dataset):
    def __init__(self, x0_dataset : Dataset, x1_dataset : Dataset):
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        
    def __len__(self):
        return min(len(self.x0_dataset), len(self.x1_dataset))
    
    def __getitem__(self, idx):
        direction = 'forward' if idx % 2 == 0 else 'backward'
        x0 = self.x0_dataset[idx]
        x1 = self.x1_dataset[idx]
        return x0, x1, direction
    
        
class ESDSBDatamodule(LightningDataModule):
    def __init__(
        self,
        x0_dataset : Dataset,
        x1_dataset : Dataset | None,
        batch_size : int,
        num_workers : int,
        x0_valset : Dataset | None = None,
        x1_valset : Dataset | None = None,
    ):
        super().__init__()
        if x1_dataset is None:
            x1_dataset = NormalDataset(x0_dataset[0].shape, len(x0_dataset))
        
        x0_dataset, x0_valset = split_dataset(x0_dataset, x0_valset, 0.95)
        x1_dataset, x1_valset = split_dataset(x1_dataset, x1_valset, 0.95)
        
        self.dsbdataset = DSBDataset(x0_dataset, x1_dataset)
        self.valset = EvalDataset(x0_valset, x1_valset)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def train_dataloader(self):
        return DataLoader(
            self.dsbdataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            drop_last = True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            drop_last = True,
        )