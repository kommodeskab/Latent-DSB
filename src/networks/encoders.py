from diffusers import VQModel, AutoencoderKL
from torch import Tensor
from torch.nn import Module
from src.networks import PretrainedModel
from transformers import MimiModel, AutoFeatureExtractor

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
    latent_std : float = 1.0
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def encode(self, x : Tensor) -> Tensor: return super().encode(x).latent_dist.mean / self.latent_std
    def decode(self, h : Tensor) -> Tensor: return super().decode(self.latent_std * h).sample
    
class PretrainedVAE:
    def __new__(cls, model_id : str, **kwargs) -> Autoencoder:
        dummy_model = AutoencoderKL.from_pretrained(model_id, **kwargs)
        print("Loaded VAE model", model_id)
        dummy_model.__class__ = Autoencoder
        return dummy_model
    
class StableDiffusionXL:
    def __new__(cls):
        encoder = PretrainedVAE("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", revision=None, variant=None)
        encoder.latent_std = 8.0
        return encoder
        
class EMNISTVAE:
    def __new__(cls):
        model = PretrainedModel("300125161951", "vae")
        model.__class__ = Autoencoder
        return model
    
class Mimi(MimiModel):
    feature_extractor : AutoFeatureExtractor
    sample_rate : int
    
    def encode(self, x : Tensor) -> Tensor:
        # x is audio with shape (batch_size, 1, seq_len)
        # we have to make it into a list of lists
        assert x.size(1) == 1, "Audio should have shape (batch_size, 1, seq_len)"
        raw_audio = x.squeeze(1).tolist()
        inputs = self.feature_extractor(
            raw_audio=raw_audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        encoded = super().encode(inputs["input_values"]).audio_codes
        encoded = encoded.unsqueeze(1).float()
        encoded = (encoded - 1000.) / 500.
        return encoded
    def decode(self, h : Tensor) -> Tensor:
        h = (h * 500.) + 1000.
        h = h.long().squeeze(1)
        return super().decode(h).audio_values
    
class PretrainedMimi:
    def __new__(cls):
        model = MimiModel.from_pretrained("kyutai/mimi")
        model.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        model.sample_rate = model.feature_extractor.sampling_rate
        print("Loaded Mimi model")
        model.__class__ = Mimi
        return model