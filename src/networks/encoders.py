from diffusers import VQModel, AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor
from torch.nn import Module

class BaseEncoderDecoder(Module):
    def encode(self, x : Tensor) -> Tensor:
        pass
    
    def decode(self, h : Tensor) -> Tensor:
        pass

class VQ(VQModel):
    def __init__(
        self,
        model_id : str,
        **kwargs,
    ):
        subfolder = kwargs.pop("subfolder", "")
        super().__init__(**kwargs)
        print("Loading pretrained VQ autoencoder from stable-diffusion..")
        dummy_model : VQModel = VQModel.from_pretrained(model_id, subfolder=subfolder)
        dummy_state_dict = dummy_model.state_dict()
        self.__dict__ = dummy_model.__dict__
        self.load_state_dict(dummy_state_dict)
        print("..done!")
        
    def encode(self, x : Tensor):
        return super().encode(x).latents
    
    def decode(self, h : Tensor):
        return super().decode(h).sample
    
class CelebAVQ(VQ):
    def __init__():
        super().__init__(model_id='CompVis/ldm-celebahq-256', subfolder='unet')
    
class IdentityEncoderDecoder(Module):
    def __init__(self):
        super().__init__()
    
    def encode(self, x : Tensor):
        return x
    
    def decode(self, h : Tensor):
        return h
    
class StableDiffusionVAE(Module):
    def __init__(
        self,
        version : str = "3.5",
    ):
        super().__init__()
        self.vae : AutoencoderKL = AutoencoderKL.from_pretrained(f"stable-diffusion-vae/{version}")
        self.vae.eval()
        
    def encode(self, x : Tensor):
        return self.vae.encode(x).latent_dist.mean
    
    def decode(self, h : Tensor):
        return self.vae.decode(h).sample