from diffusers import UNet1DModel
from torch import Tensor

class HuggingfaceUNet1D(UNet1DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        timesteps = timesteps * 1000.0
        return super().forward(x, timesteps).sample