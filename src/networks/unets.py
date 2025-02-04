from diffusers import UNet2DModel, UNet1DModel
import torch
from torch.nn import Module
from torch import Tensor

class UNet2D(UNet2DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample
        
class PretrainedUNet2D:
    def __new__(
        cls,
        model_id : str,
        **kwargs,
    ):
        subfolder = kwargs.pop("subfolder", "")
        dummy_model : UNet2DModel = UNet2DModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)
        dummy_model.__class__ = UNet2D
        return dummy_model
    
class CelebAUNet2D:
    def __new__(cls):
        return PretrainedUNet2D("CompVis/ldm-celebahq-256", subfolder="unet")
            
class EMNISTUNet2D(UNet2D):
    def __init__(self):
        super().__init__(
            in_channels=1,
            out_channels=1,
            down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D"],
            up_block_types=["UpBlock2D", "UpBlock2D", "UpBlock2D"],
            block_out_channels=[32, 32, 32],
            norm_num_groups=32,
            sample_size=4,
        )

class UNet1D(UNet1DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample

if __name__ == "__main__":
    model = PretrainedUNet2D("CompVis/ldm-celebahq-256", keep_structure=True, subfolder="unet")