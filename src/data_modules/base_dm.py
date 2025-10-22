import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import os
from typing import Optional

def split_dataset(
    train_dataset : Dataset, 
    val_dataset : Optional[Dataset], 
    train_val_split : float
    ) -> tuple[Dataset, Dataset]:
    if val_dataset is None:
        return random_split(train_dataset, [train_val_split, 1 - train_val_split])
    else:
        return train_dataset, val_dataset

class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset : Dataset,
        val_dataset : Optional[Dataset] = None,
        train_val_split : Optional[float] = None,
        **kwargs
        ):
        """
        A base data module for datasets. 
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "val_dataset"])
        self.original_dataset = dataset
        self.train_dataset, self.val_dataset = split_dataset(dataset, val_dataset, train_val_split)
        self.num_workers = kwargs.pop("num_workers", os.cpu_count())
        print(f"Using {self.num_workers} workers for data loading.")
        self.kwargs = kwargs
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset, 
            shuffle = True, 
            drop_last=True,
            num_workers=self.num_workers,
            **self.kwargs,
            )
        
    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_dataset, 
            shuffle = False, 
            drop_last=True,
            num_workers=self.num_workers,
            **self.kwargs,
            )
        
class RandomPairDataset(Dataset):
    def __init__(self, x0_dataset : Dataset, x1_dataset : Dataset):
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
    
    def __len__(self):
        return len(self.x0_dataset)
    
    def shuffle_x1_indices(self):
        print("Shuffling x1 dataset indices for random pairing...", flush=True)
        self.x1_indices = torch.randperm(len(self.x1_dataset))
    
    def __getitem__(self, idx):
        assert hasattr(self, "x1_indices"), "Please shuffle the x1 indices before getting items."
        x1_idx = self.x1_indices[idx % len(self.x1_dataset)]
        
        x0_sample = self.x0_dataset[idx]
        x1_sample = self.x1_dataset[x1_idx]
        
        return x0_sample, x1_sample

class FlowMatchingDM(BaseDM):
    train_dataset : RandomPairDataset
    
    def __init__(
        self, 
        x0_trainset : Dataset, 
        x1_trainset : Dataset, 
        x0_valset : Optional[Dataset], 
        x1_valset : Optional[Dataset], 
        train_val_split : Optional[float] = None, 
        **kwargs
        ):
        self.x0_trainset, self.x0_valset = split_dataset(x0_trainset, x0_valset, train_val_split)
        self.x1_trainset, self.x1_valset = split_dataset(x1_trainset, x1_valset, train_val_split)
        
        train_dataset = RandomPairDataset(self.x0_trainset, self.x1_trainset)
        val_dataset = RandomPairDataset(self.x0_valset, self.x1_valset)
        val_dataset.shuffle_x1_indices()
        
        super().__init__(train_dataset, val_dataset, pin_memory=False, **kwargs)
        self.original_dataset = x0_trainset
        
    def train_dataloader(self):
        self.train_dataset.shuffle_x1_indices()
        return super().train_dataloader()