from diffusers import UNet2DModel, UNet1DModel
import torch

class UNet2D(UNet2DModel):
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor): 
        return super().forward(x, time_step).sample
        
class PretrainedUNet2D:
    def __new__(cls, model_id : str, **kwargs,):
        subfolder = kwargs.pop("subfolder", "")
        dummy_model : UNet2DModel = UNet2DModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)
        dummy_model.__class__ = UNet2D
        return dummy_model
    
class CelebAUNet2D:
    def __new__(cls): 
        return PretrainedUNet2D("CompVis/ldm-celebahq-256", subfolder="unet", revision=None, variant=None)

class EMNISTUNet(UNet2D):
    def __init__(self, **kwargs):
        args = {
            "in_channels": 1,
            "out_channels": 1,
            "sample_size": 16,
            "down_block_types": ["DownBlock2D", "DownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "UpBlock2D", "UpBlock2D"],
            "block_out_channels": [32, 32, 64],
            "dropout": 0.1,
        }
        args.update(kwargs)
        super().__init__(**args)

class SmallUNet(UNet2D):
    def __init__(self, **kwargs):
        args = {
            "in_channels": 4,
            "out_channels": 4,
            "sample_size": 16,
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            "block_out_channels": [192, 256, 384, 512],
            "dropout": 0.1,
        }
        args.update(kwargs)
        super().__init__(**args)

class MediumUNet(UNet2D):
    def __init__(self, **kwargs):
        args = {
            "in_channels": 4,
            "out_channels": 4,
            "sample_size": 16,
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            "block_out_channels": [256, 384, 512, 640],
            "dropout": 0.1,
        }
        args.update(kwargs)
        super().__init__(**args)

class LargeUNet(UNet2D):
    def __init__(self, **kwargs):
        args = {
            "in_channels": 4,
            "out_channels": 4,
            "sample_size": 16,
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            "block_out_channels": [384, 512, 640, 768],
            "dropout": 0.1,
        }
        args.update(kwargs)
        super().__init__(**args)

class UNet50(UNet2D):
    def __init__(self, **kwargs):
        args = {
            "in_channels": 4,
            "out_channels": 4,
            "sample_size": 8,
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            "block_out_channels": [128, 192, 256, 384],
            "dropout": 0.1,
        }
        args.update(kwargs)
        super().__init__(**args)

class UNet1D(UNet1DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor): 
        return super().forward(x, time_step).sample
    
if __name__ == "__main__":
    unet = UNet50()
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")