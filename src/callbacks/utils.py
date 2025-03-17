from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import pytorch_lightning as pl
import torch

def get_batch_from_dataset(dataset : Dataset, batch_size : int, shuffle : bool = False) -> Tensor:
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return next(iter(dataloader))

def get_batches(trainer : pl.Trainer, batch_size : int, shuffle : bool = False) -> tuple[Tensor, Tensor]:
    start_dataset = trainer.datamodule.start_dataset_val
    end_dataset = trainer.datamodule.end_dataset_val
    
    x0 = get_batch_from_dataset(start_dataset, batch_size, shuffle)
    x1 = get_batch_from_dataset(end_dataset, batch_size, shuffle)
    
    return x0, x1

def calculate_KAD(generated : Tensor, real : Tensor, alfa : float = 100.) -> float:
    """
    This function calculates the Kernel Audio Distance between two batches of audio.
    This is the same as the Maximum Mean Discrepancy. The function uses the Gaussian kernel.
    KAD = alfa * MMD(generated, real)
    
    with MMD(x, y) = 1 / n(n-1) * sum_{i != j} k(x_i, x_j) + 1 / m(m-1) * sum_{i != j} k(y_i, y_j) - 2 / (n * m) * sum_{i, j} k(x_i, y_j)
    """
    def kernel(x : Tensor, y : Tensor) -> Tensor:
        return torch.exp(-torch.norm(x - y) ** 2)
    
    n = generated.size(0)
    m = real.size(0)
    
    Kxx = torch.zeros(n, n)
    Kyy = torch.zeros(m, m)
    Kxy = torch.zeros(n, m)
    
    for i in range(n):
        for j in range(n):
            Kxx[i, j] = kernel(generated[i], generated[j])
            
    for i in range(m):
        for j in range(m):
            Kyy[i, j] = kernel(real[i], real[j])
            
    for i in range(n):
        for j in range(m):
            Kxy[i, j] = kernel(generated[i], real[j])
            
    Kxx = Kxx.sum() / (n * (n - 1))
    Kyy = Kyy.sum() / (m * (m - 1))
    Kxy = Kxy.sum() / (n * m)
    
    return alfa * (Kxx + Kyy - 2 * Kxy).item()

if __name__ == "__main__":
    x = torch.randn(10, 1, 32, 32)
    y = torch.randn(10, 1, 32, 32)
    print(calculate_KAD(x, y))
    y = x + torch.randn_like(x) * 0.01
    print(calculate_KAD(x, y))
    
    
    
