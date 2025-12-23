import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Downsample1d(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True, out_channels: Optional[int] = None, factor: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.factor = factor
        self.op: nn.Module
        if use_conv:
            self.op = nn.Conv1d(
                self.channels, self.out_channels, 3, stride=factor, padding=1
            )
        else:
            self.op = nn.AvgPool1d(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class Upsample1d(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True, out_channels: Optional[int] = None, factor: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.factor = factor
        self.conv: Optional[nn.Conv1d] = None
        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        if self.use_conv and self.conv is not None:
            x = self.conv(x)
        return x

class ResBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Calculate "same" padding
        # For odd kernel sizes: padding = (k - 1) / 2
        padding = (kernel_size - 1) // 2

        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(8, self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        self.skip_connection: nn.Module
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h

class AttentionBlock1d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, l)
        k = k.reshape(b, self.num_heads, -1, l)
        v = v.reshape(b, self.num_heads, -1, l)

        scale = 1.0 / math.sqrt(c // self.num_heads)
        weight = torch.einsum("bhcl,bhcs->bhls", q, k) * scale
        weight = torch.softmax(weight, dim=-1)

        a = torch.einsum("bhls,bhcs->bhcl", weight, v)
        a = a.reshape(b, c, l)
        
        return x + self.proj_out(a)

class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Union[Tuple[int, ...], List[int]],
        dropout: float = 0,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 1,
        num_classes: Optional[int] = None,
        resblock_kernel_size: Union[int, List[int], Tuple[int, ...]] = 3, 
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Handle variable kernel sizes
        if isinstance(resblock_kernel_size, int):
            self.kernel_sizes = [resblock_kernel_size] * len(channel_mult)
        else:
            if len(resblock_kernel_size) != len(channel_mult):
                raise ValueError("len(resblock_kernel_size) must equal len(channel_mult)")
            self.kernel_sizes = resblock_kernel_size

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.label_emb = None

        # Input Block
        # We can also use the larger kernel here for the very first projection
        # Uses the kernel size associated with the first level (highest res)
        initial_kernel = self.kernel_sizes[0]
        padding = (initial_kernel - 1) // 2
        self.input_blocks = nn.ModuleList([
            nn.Conv1d(in_channels, model_channels, initial_kernel, padding=padding)
        ])
        
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # Downsample branch
        for level, mult in enumerate(channel_mult):
            current_kernel_size = self.kernel_sizes[level]
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [
                    ResBlock1d(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        kernel_size=current_kernel_size 
                    )
                ]
                ch = int(mult * model_channels)
                
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock1d(
                            ch,
                            num_heads=num_heads
                        )
                    )
                
                self.input_blocks.append(nn.ModuleList(layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.ModuleList([
                        Downsample1d(
                            ch, conv_resample, out_channels=out_ch
                        )
                    ])
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle block
        # Typically operates at the lowest resolution, so we use the kernel size of the last level
        mid_kernel_size = self.kernel_sizes[-1]
        self.middle_block = nn.ModuleList([
            ResBlock1d(
                ch,
                time_embed_dim,
                dropout,
                kernel_size=mid_kernel_size
            ),
            AttentionBlock1d(
                ch,
                num_heads=num_heads
            ),
            ResBlock1d(
                ch,
                time_embed_dim,
                dropout,
                kernel_size=mid_kernel_size
            ),
        ])
        self._feature_size += ch

        # Upsample branch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            current_kernel_size = self.kernel_sizes[level]
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock1d(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        kernel_size=current_kernel_size
                    )
                ]
                ch = int(model_channels * mult)
                
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock1d(
                            ch,
                            num_heads=num_heads
                        )
                    )
                
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample1d(ch, conv_resample, out_channels=out_ch)
                    )
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            SiLU(),
            zero_module(nn.Conv1d(ch, out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        hs: List[torch.Tensor] = []
        
        emb = self.time_embed(timestep_embedding(timesteps * 1000.0, self.model_channels))

        if self.num_classes is not None:
            if y is None:
                raise ValueError("Model is class-conditional but no labels (y) were provided.")
            if self.label_emb is not None:
                 emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock1d):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        for module in self.middle_block:
            if isinstance(module, ResBlock1d):
                h = module(h, emb)
            else:
                h = module(h)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock1d):
                    h = layer(h, emb)
                else:
                    h = layer(h)
        
        return self.out(h)

# --- Example Usage ---
if __name__ == "__main__":
    # Settings
    B, C, L = 8, 256, 1024
    NUM_CLASSES = 2

    model = UNet1D(
        in_channels=C,
        model_channels=512,      
        out_channels=C,
        num_res_blocks=1,
        attention_resolutions=(2, 4, 8),
        channel_mult=(1, 1, 1, 1),
        num_heads=8,
        num_classes=NUM_CLASSES,
        resblock_kernel_size=3   # Different kernel for each depth level
    )
    
    x = torch.randn(B, C, L)
    t = torch.rand((B, ))
    y = torch.randint(0, NUM_CLASSES, (B,))
    
    out = model(x, t, y=y)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")