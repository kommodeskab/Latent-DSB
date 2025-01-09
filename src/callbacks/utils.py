from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import pytorch_lightning as pl

def get_batch_from_dataset(dataset : Dataset, batch_size : int, shuffle : bool = False) -> Tensor:
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return next(iter(dataloader))

def get_batches(trainer : pl.Trainer, batch_size : int, shuffle : bool = False) -> tuple[Tensor, Tensor]:
    start_dataset = trainer.datamodule.start_dataset_val
    end_dataset = trainer.datamodule.end_dataset_val
    
    x0 = get_batch_from_dataset(start_dataset, batch_size, shuffle)
    x1 = get_batch_from_dataset(end_dataset, batch_size, shuffle)
    
    return x0, x1