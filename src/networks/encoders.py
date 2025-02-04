from diffusers import VQModel, AutoencoderKL
from torch import Tensor
from torch.nn import Module
from src.networks import PretrainedModel

class BaseEncoderDecoder(Module):
    def encode(self, x : Tensor) -> Tensor:
        pass
    
    def decode(self, h : Tensor) -> Tensor:
        pass
    
class IdentityEncoderDecoder(Module):
    def __init__(self):
        super().__init__()

    def encode(self, x : Tensor) -> Tensor:
        return x
    
    def decode(self, h : Tensor) -> Tensor:
        return h
    
class VQ(VQModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def encode(self, x : Tensor) -> Tensor:
        return super().encode(x).latents
    
    def decode(self, h : Tensor) -> Tensor:
        return super().decode(h).sample
    
class PretrainedVQ:
    def __new__(cls, model_id : str, **kwargs):
        subfolder = kwargs.pop("subfolder", "")
        dummy_model = VQModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)
        dummy_model.__class__ = VQ
        return dummy_model
    
class CelebAVQ:
    def __new__(cls):
        return PretrainedVQ("CompVis/ldm-celebahq-256", subfolder="vqvae")
        
class AutoencoderKLMixin:
    @staticmethod
    def wrap(model : AutoencoderKL):
        model.__encode = model.encode
        model.__decode = model.decode
        
        def encode(x : Tensor):
            return model.__encode(x).latent_dist.mean
        
        def decode(h : Tensor):
            return model.__decode(h).sample
        
        setattr(model, "encode", encode)
        setattr(model, "decode", decode)
    
        return model
        
class EMNISTVAE(AutoencoderKLMixin):
    def __new__(cls):
        model = PretrainedModel("vae", "300125161951", "vae")
        print("Loaded pretrained VAE encoder/decoder")
        return super().wrap(model)
    
class StableDiffusionVAE(AutoencoderKLMixin):
    def __new__(cls, version : str = "3.5"):
        model = AutoencoderKL.from_pretrained(f"stable-diffusion-vae/{version}")
        print(f"Loaded stable-diffusion-vae/{version}")
        return super().wrap(model)