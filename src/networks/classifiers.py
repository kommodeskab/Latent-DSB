from torch.nn import Module
import torch.nn as nn
from torch import Tensor

class EMNISTClassifier(Module):
    def __init__(self):
        super().__init__()
        hidden_sizes = [256, 8]
        self.fc1 = nn.Linear(16 * 16, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def common_forward(self, x : Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x
        
    def forward(self, x : Tensor) -> Tensor:
        x = self.common_forward(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def h(self, x : Tensor) -> Tensor:
        x = self.common_forward(x)
        return x