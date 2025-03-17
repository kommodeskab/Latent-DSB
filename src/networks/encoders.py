from diffusers import VQModel, AutoencoderKL
from torch import Tensor
from torch.nn import Module
from src.networks import PretrainedModel
from transformers import MimiModel, AutoFeatureExtractor
from torch import Tensor
import torch

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

class Mimi(MimiModel):
    feature_extractor : AutoFeatureExtractor
    sample_rate : int
    old_range = (0., 2047.)
    new_range = (-1024., 1024.)
    
    @staticmethod
    def normalize(x : Tensor, old_range : tuple, new_range : tuple) -> Tensor:
        old_min, old_max = old_range
        new_min, new_max = new_range
        return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    
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
        encoded = self.normalize(encoded, self.old_range, self.new_range)
        return encoded
    
    def decode(self, h : Tensor) -> Tensor:
        h = self.normalize(h, self.new_range, self.old_range)
        h = h.round().squeeze(1).clamp(*self.old_range).long()
        return super().decode(h).audio_values

class PretrainedMimi:
    def __new__(cls):
        model = MimiModel.from_pretrained("kyutai/mimi")
        model.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        model.sample_rate= model.feature_extractor.sampling_rate
        print("Loaded Mimi model")
        model.__class__ = Mimi
        return model
    
class STFTEncoderDecoder(Module):
    def __init__(self, n_fft : int, hop_length : int, win_length : int):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
    def encode(self, audio : Tensor) -> Tensor:
        audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=False
        )
        spec = spec.permute(0, 3, 1, 2)
        return spec
    
    def decode(self, spec : Tensor) -> Tensor:
        spec = spec.permute(0, 2, 3, 1)
        real, imag = spec[..., 0], spec[..., 1]
        spec = torch.complex(real, imag)
        
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=False
        )
        return audio