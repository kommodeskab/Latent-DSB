from src.lightning_modules.esdsb import DIRECTIONS
from torch.utils.data import Dataset, IterableDataset
from torch import Tensor
import random
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data_modules.base_dm import split_dataset
import multiprocessing as mp
from multiprocessing.managers import SyncManager, ListProxy
from typing import Generator

class Cache:
    cache : ListProxy
    
    def clear(self) -> None: 
        self.cache[:] = []
        
    def is_empty(self) -> bool: 
        return len(self.cache) == 0

    def add(self, x0 : Tensor, x1 : Tensor, direction : DIRECTIONS) -> None:
        batch_size = x0.shape[0]
        for i in range(batch_size):
            x0_sample, x1_sample = x0[i], x1[i]
            self.cache.append((x0_sample, x1_sample, direction, True)) 
            
class DSBCache(IterableDataset, Cache):
    def __init__(self, manager : SyncManager):
        super().__init__()
        self.cache = manager.list()

    def __iter__(self) -> Generator[tuple[Tensor, Tensor, str, bool], None, None]:
        # return random element from cache
        while True:
            idx = random.randint(0, len(self.cache) - 1)
            yield self.cache[idx]
            
class EvalCache(Dataset, Cache):
    def __init__(self, manager : SyncManager):
        super().__init__()
        self.cache = manager.list()

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx : int) -> tuple[Tensor, Tensor, str, bool]:
        return self.cache[idx]
        
class DSBDataset(IterableDataset):
    def __init__(
        self, 
        x0_dataset : Dataset[Tensor],
        x1_dataset : Dataset[Tensor],
        ):
        super().__init__()
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        
    def __iter__(self) -> Generator[tuple[Tensor, Tensor, str, bool], None, None]:
        while True:
            direction = 'forward' if random.random() < 0.5 else 'backward'
            x0_idx = random.randint(0, len(self.x0_dataset) - 1)
            x1_idx = random.randint(0, len(self.x1_dataset) - 1)
            x0 = self.x0_dataset[x0_idx]
            x1 = self.x1_dataset[x1_idx]
            yield x0, x1, direction, False

class EvalDataset(Dataset):
    def __init__(
        self,
        x0_dataset : Dataset[Tensor],
        x1_dataset : Dataset[Tensor],
    ):
        super().__init__()
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        
    def __len__(self) -> int:
        return min(len(self.x0_dataset), len(self.x1_dataset))

    def __getitem__(self, idx : int) -> tuple[Tensor, Tensor, str, bool]:
        direction = 'forward' if idx % 2 == 0 else 'backward'
        x0 = self.x0_dataset[idx]
        x1 = self.x1_dataset[idx]
        return x0, x1, direction, False

class ESDSBDatamodule(LightningDataModule):    
    def __init__(
        self,
        x0_dataset : Dataset,
        x1_dataset : Dataset,
        batch_size : int,
        num_workers : int,
        x0_valset : Dataset | None = None,
        x1_valset : Dataset | None = None,
    ):
        super().__init__()
        
        x0_dataset, x0_valset = split_dataset(x0_dataset, x0_valset, 0.95)
        x1_dataset, x1_valset = split_dataset(x1_dataset, x1_valset, 0.95)
        
        print(f"Training set sizes: x0={len(x0_dataset)}, x1={len(x1_dataset)}")
        print(f"Validation set sizes: x0={len(x0_valset)}, x1={len(x1_valset)}")
        
        self.dataset = DSBDataset(x0_dataset, x1_dataset)
        self.valset = EvalDataset(x0_valset, x1_valset)
        
        manager = mp.Manager()
        self.cache = DSBCache(manager)
        self.val_cache = EvalCache(manager)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def train_dataloader(self):
        dataset = self.dataset if self.cache.is_empty() else self.cache
        
        return DataLoader(
            dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory= True,
            drop_last = True,
            persistent_workers=False,
        )
        
    def val_dataloader(self):
        valset = self.valset if self.val_cache.is_empty() else self.val_cache
        
        return DataLoader(
            valset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory= True,
            drop_last = True,
            persistent_workers=False,
        )