import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import random

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
        batch_size : int = 32,
        num_workers: int = 4,
        ):
        """
        A base data module for datasets. 
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "val_dataset"])
        
        self.dataset = dataset
        self.train_dataset, self.val_dataset = split_dataset(dataset, val_dataset, train_val_split)
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset, 
            shuffle = True, 
            drop_last=True,
            persistent_workers=True,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory=True
            )
        
    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_dataset, 
            shuffle = False, 
            drop_last=True,
            persistent_workers=True,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory=True
            )
        
class RandomPairDataset(Dataset):
    def __init__(self, x0_dataset : Dataset, x1_dataset : Dataset):
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset
        
    def __len__(self):
        return min(len(self.x0_dataset), len(self.x1_dataset))
    
    def __getitem__(self, idx):
        # randomly sample from both datasets
        x0_rand_idx = random.randint(0, len(self.x0_dataset) - 1)
        x1_rand_idx = random.randint(0, len(self.x1_dataset) - 1)
        return self.x0_dataset[x0_rand_idx], self.x1_dataset[x1_rand_idx]

class FlowMatchingDM(BaseDM):
    def __init__(self, x0_dataset : Dataset, x1_dataset : Dataset, x0_dataset_val : Dataset | None = None, x1_dataset_val : Dataset | None = None, **kwargs):
        assert (x0_dataset_val is None) == (x1_dataset_val is None), "Validation datasets must either both be None or both not be None."
        paired_dataset = RandomPairDataset(x0_dataset, x1_dataset)
        
        if x0_dataset_val is not None:
            paired_dataset_val = RandomPairDataset(x0_dataset_val, x1_dataset_val)
        else:
            paired_dataset_val = None
            
        super().__init__(paired_dataset, paired_dataset_val, **kwargs)

class BaseDSBDM(pl.LightningDataModule):
    def __init__(
        self,
        start_dataset : Dataset,
        end_dataset : Dataset,
        start_dataset_val : Dataset | None = None,
        end_dataset_val : Dataset | None = None,
        train_val_split : float = 0.95,
        batch_size : int = 32,
        num_workers: int = 4,
        ):
        """
        A special datamodule for the DSB algorithm. 
        It takes two datasets, one for the start and one for the end, and splits them into train and validation.
        It returns end_dataset when training forward and start_dataset when training backward.
        Also uses the special CacheDataLoader for the cache implementation in the DSB algorithm.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["start_dataset", "end_dataset", "start_dataset_val", "end_dataset_val"])
        
        self.start_dataset = start_dataset
        self.end_dataset = end_dataset
        
        self.start_dataset_train, self.start_dataset_val = split_dataset(start_dataset, start_dataset_val, train_val_split)
        self.end_dataset_train, self.end_dataset_val = split_dataset(end_dataset, end_dataset_val, train_val_split)
        
        self.loader_kwargs = {
            "batch_size" : batch_size,
            "num_workers" : num_workers,
            "persistent_workers" : True,
            "drop_last" : True,
            "pin_memory" : False
        }
        
        self.training_backward = None
    
    def train_dataloader(self):
        assert self.training_backward is not None, "Please set the training_backward attribute before calling train_dataloader."
        training_backward = self.training_backward
        dataset = self.start_dataset_train if training_backward else self.end_dataset_train
        return DataLoader(
            dataset = dataset, 
            shuffle = True,
            **self.loader_kwargs
            )
        
    def val_dataloader(self):
        return [
            DataLoader(dataset = self.start_dataset_val, shuffle = False, **self.loader_kwargs),
            DataLoader(dataset = self.end_dataset_val, shuffle = False, **self.loader_kwargs)
        ]