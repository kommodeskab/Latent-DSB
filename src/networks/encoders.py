from diffusers import VQModel, AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor
from torch.nn import Module
from src.networks import PretrainedModel

class BaseEncoderDecoder(Module):
    def encode(self, x : Tensor) -> Tensor:
        pass
    
    def decode(self, h : Tensor) -> Tensor:
        pass
    
class IdentityEncoderDecoder:
    def encode(self, x : Tensor):
        return x
    
    def decode(self, h : Tensor):
        return h
    
class VQ:
    def __new__(cls, model_id : str, **kwargs):
        subfolder = kwargs.pop("subfolder", "")
        model = VQModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)
        print(f"Loaded a VQ model with id {model_id}")
        return cls.wrap(model)
    
    @staticmethod
    def wrap(model : VQModel):
        model.__encode = model.encode
        model.__decode = model.decode
        
        def encode(x : Tensor):
            return model.__encode(x).latents
        
        def decode(h : Tensor):
            return model.__decode(h).sample
        
        setattr(model, "encode", encode)
        setattr(model, "decode", decode)
        
        return model
    
class CelebAVQ:
    def __new__(cls):
        return VQ("CompVis/ldm-celebahq-256", subfolder="vqvae")
        
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