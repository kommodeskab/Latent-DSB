import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLinearConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int, 
        hidden_dim: int,
        n_layers: int = 2,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.n_layers = n_layers

        layers = []
        for i in range(n_layers):
            in_dim = in_channels * kernel_size if i == 0 else hidden_dim
            out_dim = out_channels if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode='reflect')

        patches = F.unfold(
            x.unsqueeze(-1), 
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=0
        ) 

        patches = patches.transpose(1, 2)
        out = self.mlp(patches)
        out = out.transpose(1, 2)

        return out
    
class NonLinearConv1dTranspose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int, 
        hidden_dim: int,
        n_layers: int = 2,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.n_layers = n_layers

        layers = []
        mlp_in_dim = in_channels
        mlp_out_dim = out_channels * kernel_size

        for i in range(n_layers):
            current_in_dim = mlp_in_dim if i == 0 else hidden_dim
            current_out_dim = mlp_out_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(current_in_dim, current_out_dim))
            if i < n_layers - 1:
                layers.append(activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_out, L_out = x.shape

        patches = x.transpose(1, 2)
        unfolded_data = self.mlp(patches)
        unfolded_data = unfolded_data.transpose(1, 2) 
        L_in_padded = (L_out - 1) * self.stride + self.kernel_size
         
        folded_data_sum = F.fold(
            unfolded_data,
            output_size=(L_in_padded, 1),
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
        ).squeeze(-1) # -> (B, C_in * K, L_in_padded)
        weights = torch.ones(1, 1 * self.kernel_size, L_out, device=x.device)
        
        overlap_counts = F.fold(
            weights, 
            output_size=(L_in_padded, 1),
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
        ).squeeze(-1) # -> (1, 1 * K, L_in_padded)
        
        overlap_counts_repeated = overlap_counts.repeat(1, self.out_channels * self.kernel_size, 1)
        overlap_counts_repeated[overlap_counts_repeated == 0] = 1
        output_averaged = (folded_data_sum / overlap_counts_repeated).contiguous() # (B, C_in * K, L_in_padded)
        
        output_padded = output_averaged.view(
            B, 
            self.out_channels, 
            self.kernel_size, 
            L_in_padded
        ).sum(dim=2) # Simple summation over the kernel dimension

        start = self.pad
        end = output_padded.shape[-1] - self.pad
        
        out = output_padded[:, :, start:end].contiguous()
        
        return out