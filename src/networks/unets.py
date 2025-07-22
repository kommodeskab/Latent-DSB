from diffusers import UNet2DModel, UNet1DModel
import torch
from torch import Tensor
from torch.nn import Embedding

class UNet2D(UNet2DModel):
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

    def forward(self, x : Tensor, timestep : Tensor, class_labels : Tensor = None) -> Tensor: 
        if torch.is_floating_point(timestep):
            timestep = (timestep * 1000).int()
                
        return super().forward(x, timestep, class_labels).sample

class UNet1D(UNet1DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x : Tensor, time_step : Tensor) -> Tensor:
        return super().forward(x, time_step).sample
        
class UNet1D50(UNet1D):
    def __init__(self, **kwargs):
        args = {
            "block_out_channels": [256, 320, 384, 384],
            "extra_in_channels": 16,
            "down_block_types": ["DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D"],
            "up_block_types": ["AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"],
        }
        args.update(kwargs)
        super().__init__(**args)