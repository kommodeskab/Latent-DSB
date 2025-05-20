import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import os
from rich.console import Console
from rich.panel import Panel

def split_dataset(train_dataset : Dataset, val_dataset : Dataset | None, train_val_split : float) -> tuple[Dataset, Dataset]:
    if val_dataset is None:
        return random_split(train_dataset, [train_val_split, 1 - train_val_split])
    else:
        return train_dataset, val_dataset

class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset : Dataset,
        val_dataset : Dataset | None = None,
        train_val_split : float = 0.95,
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
        self.x1_indices = torch.randperm(len(self.x1_dataset))
    
    def __getitem__(self, idx):
        assert hasattr(self, "x1_indices"), "Please shuffle the x1 indices before getting items."
        x0_sample = self.x0_dataset[idx]
        x1_idx = self.x1_indices[idx % len(self.x1_dataset)]
        x1_sample = self.x1_dataset[x1_idx]
        
        return x0_sample, x1_sample

class FlowMatchingDM(BaseDM):
    train_dataset : RandomPairDataset
    
    def __init__(self, x0_dataset : Dataset, x1_dataset : Dataset, x0_dataset_val : Dataset | None = None, x1_dataset_val : Dataset | None = None, train_val_split : float = 0.95, flip : bool = False, **kwargs):
        if flip:
            x0_dataset, x1_dataset = x1_dataset, x0_dataset
            x0_dataset_val, x1_dataset_val = x1_dataset_val, x0_dataset_val
            
        x0_train, x0_val = split_dataset(x0_dataset, x0_dataset_val, train_val_split)
        x1_train, x1_val = split_dataset(x1_dataset, x1_dataset_val, train_val_split)
        
        train_dataset = RandomPairDataset(x0_train, x1_train)
        val_dataset = RandomPairDataset(x0_val, x1_val)
        val_dataset.shuffle_x1_indices()
        
        super().__init__(train_dataset, val_dataset, pin_memory=False, **kwargs)
        console = Console()
        console.print(
            Panel.fit(
                "[bold red] ⚠ Remember to use 'reload_dataloaders_every_n_epochs=1' in your Trainer ⚠",
                title="Using Flow Matching Data Module",
                border_style="red",
            )
        )
        self.original_dataset = x0_dataset
        
    def train_dataloader(self):
        self.train_dataset.shuffle_x1_indices()
        return super().train_dataloader()