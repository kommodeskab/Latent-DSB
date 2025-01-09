from diffusers import UNet2DModel, UNet1DModel
import torch

class UNet2D(UNet2DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample
        
class PretrainedUNet2D(UNet2D):
    def __init__(
        self,
        model_id : str,
        **kwargs,
    ):
        super().__init__()
        dummy_model : torch.nn.Module = UNet2DModel.from_pretrained(pretrained_model_name_or_path = model_id, **kwargs)
        self.__dict__ = dummy_model.__dict__.copy()
        self.load_state_dict(dummy_model.state_dict())
    
class UNet1D(UNet1DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample

if __name__ == "__main__":
    model = PretrainedUNet2D("google/ddpm-cifar10-32")