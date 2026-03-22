import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AudioConvStem(nn.Module):
    """
    Hierarchically downsamples raw audio into Transformer tokens.
    Example total stride: 4 * 4 * 4 = 64.
    """
    def __init__(self, in_channels: int, model_size: int) -> None:
        super().__init__()
        # Gradually increase channels while downsampling
        self.kernel_size = 16
        self.stride = 4
        self.padding = 6
        
        hidden = model_size // 2
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.GroupNorm(num_groups=8, num_channels=hidden),
            nn.GELU(approximate="tanh"),
            
            nn.Conv1d(hidden, hidden, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.GroupNorm(num_groups=8, num_channels=hidden),
            nn.GELU(approximate="tanh"),
            
            nn.Conv1d(hidden, model_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
    def calculate_output_len(self, input_len: int) -> int:
        l_out = input_len
        for _ in range(3):  # Assuming 3 layers
            l_out = (l_out + 2 * self.padding - self.kernel_size) // self.stride + 1
        return l_out


class AudioConvHead(nn.Module):
    """
    Hierarchically upsamples Transformer tokens back to raw audio.
    Symmetrical to the Stem.
    """
    def __init__(self, model_size: int, out_channels: int) -> None:
        super().__init__()
        self.kernel_size = 16
        self.stride = 4
        self.padding = 6
        
        hidden = model_size // 2
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(model_size, hidden, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.GroupNorm(num_groups=8, num_channels=hidden),
            nn.GELU(approximate="tanh"),
            
            nn.ConvTranspose1d(hidden, hidden, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.GroupNorm(num_groups=8, num_channels=hidden),
            nn.GELU(approximate="tanh"),
            
            nn.ConvTranspose1d(hidden, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


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


class DiTBlock(nn.Module):
    def __init__(self, model_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(model_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(model_size, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(model_size, model_size * 4), 
            nn.GELU(approximate="tanh"), 
            nn.Dropout(dropout),
            nn.Linear(model_size * 4, model_size)
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_size, 6 * model_size, bias=True))
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x, need_weights=False)
        x = x + self.dropout(gate_msa.unsqueeze(1) * attn_out)

        normed_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(normed_x2)
        x = x + self.dropout(gate_mlp.unsqueeze(1) * mlp_out)

        return x


class AudioDiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_seq_len: int,
        in_channels: int,
        out_channels: int,
        model_size: int,
        num_classes: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_size = model_size

        self.encoder = AudioConvStem(in_channels, model_size)
        max_seq_len = self.encoder.calculate_output_len(input_seq_len)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, model_size))
        self.pos_drop = nn.Dropout(dropout)

        self.t_embedder = TimestepEmbedder(model_size)
        self.y_embedder = nn.Embedding(num_classes, model_size)

        self.blocks = nn.ModuleList([DiTBlock(model_size, num_heads, dropout) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(model_size, elementwise_affine=False, eps=1e-6)
                
        self.decoder = AudioConvHead(model_size, out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.y_embedder.weight, std=0.02)

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self.encoder(x)
        x = x.transpose(1, 2)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        t_emb = self.t_embedder.forward(t)
        
        if y is not None:
            y_emb = self.y_embedder.forward(y)
            c = t_emb + y_emb
        else:
            c = t_emb

        for block in self.blocks:
            x = block(x, c)

        x = self.final_norm(x)
        x = x.transpose(1, 2)

        x = self.decoder(x)

        return x
    
if __name__ == "__main__":
    model = AudioDiffusionTransformer(
        input_seq_len=48000,
        in_channels=1,
        out_channels=1,
        model_size=512,
        num_classes=2,
        depth=10,
        num_heads=8,
        dropout=0.1
    )
    x = torch.randn(2, 1, 48000)  # Batch of 2, 1 channel, 48000 samples
    t = torch.rand(2)  # Random timesteps for the batch
    y = torch.randint(0, 2, (2,))  # Random class
    with torch.no_grad():
        out = model(x, t, y)
    print(out.shape)  # Should be (2, 1, 48000)