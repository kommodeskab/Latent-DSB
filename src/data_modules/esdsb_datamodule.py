from src.lightning_modules.esdsb import DIRECTIONS
from torch.utils.data import Dataset
from torch import Tensor
import random
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data_modules.base_dm import split_dataset
import multiprocessing as mp
from multiprocessing.managers import SyncManager
            
class CacheDataset(Dataset):
    def __init__(self, manager : SyncManager):
        super().__init__()
        self.cache = manager.list()
        
    def __len__(self): 
        return len(self.cache)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]: 
        return self.cache[idx]
    
    def clear(self) -> None: 
        self.cache[:] = []
        
    def is_empty(self) -> bool: 
        return len(self.cache) == 0

    def add(self, x0 : Tensor, x1 : Tensor, direction : DIRECTIONS) -> None:
        batch_size = x0.shape[0]
        for i in range(batch_size):
            x0_sample, x1_sample = x0[i], x1[i]
            self.cache.append((x0_sample, x1_sample, direction, True))   
        
class DSBDataset(Dataset):
    def __init__(
        self, 
        x0_dataset : Dataset[Tensor],
        x1_dataset : Dataset[Tensor],
        training : bool = True
        ):
        super().__init__()
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        self.training = training
        
    def get_x1_idx(self, x0_idx : int) -> int:
        if self.training:
            return random.randint(0, len(self.x1_dataset) - 1)
        else:
            return x0_idx
            
    def __getitem__(self, idx : int) -> tuple[Tensor, Tensor, str, bool]:
        direction = 'forward' if idx % 2 == 0 else 'backward'
        
        x0_idx = idx
        x1_idx = self.get_x1_idx(x0_idx)

        x0 = self.x0_dataset[x0_idx]
        x1 = self.x1_dataset[x1_idx]
        
        return x0, x1, direction, False
    
    def __len__(self) -> int:
        return len(self.x0_dataset)
        
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
        
        self.dataset = DSBDataset(x0_dataset, x1_dataset, training = True)
        self.valset = DSBDataset(x0_valset, x1_valset, training = False)
        
        manager = mp.Manager()
        self.cache = CacheDataset(manager)
        self.val_cache = CacheDataset(manager)
        
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
            persistent_workers=True if self.num_workers > 0 else False,
        )
        
    def val_dataloader(self):
        valset = self.valset if self.val_cache.is_empty() else self.val_cache
        
        return DataLoader(
            valset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            drop_last = True,
            pin_memory= True,
            persistent_workers=True if self.num_workers > 0 else False,
        )