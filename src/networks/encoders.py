from diffusers import VQModel, AutoencoderKL
from torch import Tensor
from torch.nn import Module
from src.networks import PretrainedModel

class BaseEncoderDecoder(Module):
    def encode(self, x : Tensor) -> Tensor: ...    
    def decode(self, h : Tensor) -> Tensor: ...
    
class IdentityEncoderDecoder(Module):
    def __init__(self):
        super().__init__()
    def encode(self, x : Tensor) -> Tensor: return x
    def decode(self, h : Tensor) -> Tensor: return h
    
class VQ(VQModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encode(self, x : Tensor) -> Tensor: return super().encode(x).latents
    def decode(self, h : Tensor) -> Tensor: return super().decode(h).sample
    
class PretrainedVQ:
    def __new__(cls, model_id : str, **kwargs) -> VQ:
        subfolder = kwargs.pop("subfolder", "")
        dummy_model = VQModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)
        print("Loaded VQ model", model_id)
        dummy_model.__class__ = VQ
        return dummy_model

class CelebAVQ:
    def __new__(cls): return PretrainedVQ("CompVis/ldm-celebahq-256", subfolder="vqvae", revision=None, variant=None)
    
class Autoencoder(AutoencoderKL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encode(self, x : Tensor) -> Tensor: return super().encode(x).latent_dist.mean
    def decode(self, h : Tensor) -> Tensor: return super().decode(h).sample
    
class PretrainedVAE:
    def __new__(cls, model_id : str, **kwargs) -> Autoencoder:
        dummy_model = AutoencoderKL.from_pretrained(model_id, **kwargs)
        print("Loaded VAE model", model_id)
        dummy_model.__class__ = Autoencoder
        return dummy_model
    
class StableDiffusionXL:
    def __new__(cls):
        return PretrainedVAE("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", revision=None, variant=None)
        
class EMNISTVAE:
    def __new__(cls):
        model = PretrainedModel("300125161951", "vae")
        model.__class__ = Autoencoder
        return model

class StableDiffusionVAE:
    def __new__(cls, version : str = "3.5"):
        model = AutoencoderKL.from_pretrained(f"stable-diffusion-vae/{version}")
        print(f"Loaded stable-diffusion-vae/{version}")
        model.__class__ = Autoencoder
        return model