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
        
class PretrainedUNet2D(UNet2D):
    def __init__(
        self,
        model_id : str,
        **kwargs,
    ):
        """
        Args:
            model_id (str): The model id of the pretrained model.
            keep_structure (bool): If True, the structure of the model will be kept, but the weights will be loaded from the pretrained model. If False, the weights and structure will be loaded from the pretrained model.
        """
        subfolder = kwargs.pop("subfolder", "")
        super().__init__(**kwargs)
        print("Loading pretrained model...")
        dummy_model : Module = UNet2DModel.from_pretrained(model_id, subfolder=subfolder)
        dummy_state_dict = dummy_model.state_dict()
        self.__dict__ = dummy_model.__dict__
        self.load_state_dict(dummy_state_dict)
        print("Done loading pretrained model.")

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