import math
import torch
import torch.nn as nn
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
    def __init__(self, model_size: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(model_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(model_size, elementwise_affine=False, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(model_size, model_size * 4), nn.GELU(approximate="tanh"), nn.Linear(model_size * 4, model_size)
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_size, 6 * model_size, bias=True))

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        normed_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(normed_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class AudioDiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_seq_len: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        model_size: int,
        num_classes: int,
        depth: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 0, "Kernel size should be divisible by 2 for proper downsampling/upsampling"
        padding = kernel_size // 2

        max_seq_len = (input_seq_len + 2 * padding - kernel_size) // stride + 1

        self.model_size = model_size

        self.conv_in = nn.Conv1d(
            in_channels,
            model_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
        )
        # Note: max_seq_len should accommodate the downsampled length
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, model_size))

        self.t_embedder = TimestepEmbedder(model_size)
        self.y_embedder = nn.Embedding(num_classes, model_size)

        self.blocks = nn.ModuleList([DiTBlock(model_size, num_heads) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(model_size, elementwise_affine=False, eps=1e-6)
        self.conv_out = nn.ConvTranspose1d(
            model_size, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # Zero-initialize final layer for identity function at start
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # Input shape: (B, C, L)
        x = self.conv_in(x)  # Shape: (B, model_size, L')
        x = x.transpose(1, 2)  # Shape: (B, L', model_size)

        x = x + self.pos_embed[:, : x.size(1), :]

        t_emb = self.t_embedder.forward(t)
        y_emb = self.y_embedder.forward(y)
        c = t_emb + y_emb

        for block in self.blocks:
            x = block(x, c)

        x = self.final_norm(x)
        x = x.transpose(1, 2)  # Back to (B, model_size, L')

        # Output shape: (B, C, L)
        x = self.conv_out(x)

        return x
