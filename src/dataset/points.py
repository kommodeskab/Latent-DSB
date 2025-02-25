from torch.utils.data import Dataset
import torch
import random

class Points(Dataset):
    def __init__(self, n_points : int = 50_000):
        super().__init__()
        torch.manual_seed(0)
        self.n_points = n_points
        n_modes = 2
        
        self.x0_modes = torch.randn(n_modes, 2) * 2 + 1
        self.x1_modes = torch.randn(n_modes, 2) * 1 - 1
        
        x0_points = []
        x1_points = []
        
        for _ in range(n_points):
            x0_mode = self.x0_modes[random.randint(0, n_modes - 1)]
            x1_mode = self.x1_modes[random.randint(0, n_modes - 1)]
            
            x0_points.append(x0_mode + torch.randn(2) * 0.1)
            x1_points.append(x1_mode + torch.randn(2) * 0.1)
            
        self.x0_points = torch.stack(x0_points)
        self.x1_points = torch.stack(x1_points)
        
        x1_indices = list(range(n_points))
        random.shuffle(x1_indices)
        self.x1_indices = x1_indices
        
    def __len__(self):
        return self.n_points
    
    def __getitem__(self, idx):
        x0 = self.x0_points[idx]
        x1_idx = self.x1_indices[idx % len(self.x1_points)]
        x1 = self.x1_points[x1_idx]
        return x0, x1
        
        