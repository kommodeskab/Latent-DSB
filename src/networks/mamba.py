import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2
from torch import Tensor
import math
from typing import Optional


class TimestepEmbedder(nn.Module):
    def __init__(self, model_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, model_size, bias=True),
            nn.SiLU(),
            nn.Linear(model_size, model_size, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        t = t * 1000.0
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)


class AdaLN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.scale_proj = nn.Linear(d_model, d_model)
        self.shift_proj = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: Tensor, emb: Optional[Tensor] = None) -> Tensor:
        x_norm = self.norm(x)  # (batch_size, seq_len, d_model)
        if emb is None:
            return x_norm
        scale = self.scale_proj(emb)  # (batch_size, d_model) or (batch_size, seq_len, d_model)
        shift = self.shift_proj(emb)  # (batch_size, d_model) or (batch_size, seq_len, d_model)
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x_norm * (1 + scale) + shift


class Mamba2DiffusionBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        # Standard LayerNorm (or RMSNorm)
        self.adaln = AdaLN(d_model)

        self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state)
        self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state)

        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        x_norm = self.adaln(x, emb)  # (batch_size, seq_len, d_model)

        out_fwd = self.mamba_fwd(x_norm)

        x_norm_flipped = torch.flip(x_norm, dims=[1])
        out_bwd = self.mamba_bwd(x_norm_flipped)
        out_bwd = torch.flip(out_bwd, dims=[1])

        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.out_proj(out)

        return x + out


class Mamba2DiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_model: int,
        d_state: int,
        num_blocks: int,
        num_classes: int,
        kernel_size: int = 256,
        stride: int = 16,
    ):
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.down_conv = nn.Conv1d(
            in_channels, d_model, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.up_conv = nn.ConvTranspose1d(
            d_model, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.y_embedder = nn.Embedding(num_classes, d_model)
        self.timestep_embedder = TimestepEmbedder(model_size=d_model)
        self.blocks = nn.ModuleList([Mamba2DiffusionBlock(d_model=d_model, d_state=d_state) for _ in range(num_blocks)])

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self.down_conv(x)
        x = x.transpose(1, 2)

        emb = self.timestep_embedder(t)

        if y is not None:
            emb = emb + self.y_embedder(y)

        for block in self.blocks:
            x = block(x, emb)

        x = x.transpose(1, 2)
        x = self.up_conv(x)

        return x


class Mamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.adaln = AdaLN(d_model)
        self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state)
        self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x_norm = self.adaln(x, y)

        out_fwd = self.mamba_fwd(x_norm)

        x_norm_flipped = torch.flip(x_norm, dims=[1])
        out_bwd = self.mamba_bwd(x_norm_flipped)
        out_bwd = torch.flip(out_bwd, dims=[1])

        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.out_proj(out)

        return x + out


class Mamba2Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_model: int,
        d_state: int,
        num_blocks: int,
        kernel_size: int = 256,
        stride: int = 16,
        conditional: bool = False,
    ):
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.down_conv = nn.Conv1d(
            in_channels, d_model, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        if conditional:
            self.y_down_conv = nn.Conv1d(
                in_channels, d_model, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        self.up_conv = nn.ConvTranspose1d(
            d_model, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.blocks = nn.ModuleList([Mamba2Block(d_model=d_model, d_state=d_state) for _ in range(num_blocks)])

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self.down_conv(x)
        x = x.transpose(1, 2)

        if y is not None:
            y = self.y_down_conv(y)
            y = y.transpose(1, 2)

        for block in self.blocks:
            x = block(x, y)

        x = x.transpose(1, 2)
        x = self.up_conv(x)

        return x
