from diffusers import VQModel, AutoencoderKL
from torch import Tensor
from torch.nn import Module
from transformers import MimiModel, AutoFeatureExtractor
from torch import Tensor
import torch
import dac

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
    def encode(self, x : Tensor) -> Tensor: return super().encode(x).latent_dist.sample() * self.config.scaling_factor
    def decode(self, h : Tensor) -> Tensor: return super().decode(h / self.config.scaling_factor).sample
    
class PretrainedVAE:
    def __new__(cls, model_id : str, **kwargs) -> Autoencoder:
        dummy_model = AutoencoderKL.from_pretrained(model_id, **kwargs)
        print("Loaded VAE model", model_id)
        dummy_model.__class__ = Autoencoder
        return dummy_model
    
class StableDiffusionXL:
    def __new__(cls):
        encoder = PretrainedVAE("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", revision=None, variant=None)
        encoder.eval()
        return encoder
    
def normalize(x : Tensor, old_range : tuple, new_range : tuple) -> Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

class Mimi(MimiModel):
    feature_extractor : AutoFeatureExtractor
    sample_rate : int
    old_range = (0., 2047.)
    new_range = (-20., 20.)
    
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
        encoded = normalize(encoded, self.old_range, self.new_range)
        return encoded
    
    def decode(self, h : Tensor) -> Tensor:
        h = normalize(h, self.new_range, self.old_range)
        h = h.round().squeeze(1).clamp(*self.old_range).long()
        return super().decode(h).audio_values

class PretrainedMimi:
    def __new__(cls, new_range : tuple = (-20, 20)) -> Mimi:
        model = MimiModel.from_pretrained("kyutai/mimi")
        model.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        model.sample_rate= model.feature_extractor.sampling_rate
        model.new_range = new_range
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
    
class DACEncodec(Module):
    old_range = (0., 1023.)
    new_range = (-10., 10.)
    
    def __init__(self, model_type : str = '16khz'):
        super().__init__()
        if model_type == '16khz':
            self.sample_rate = 16000
        else:
            raise NotImplementedError(f"Not implemented for {model_type} yet.")
        dac_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(dac_path)
        self.model.eval()
        self.waveform_len = None
        
    def s_to_zq(self, s : Tensor) -> Tensor:
        zq, _, _ = self.model.quantizer.from_codes(s)
        return zq
        
    @torch.no_grad()
    def encode(self, x : Tensor) -> Tensor:
        if self.waveform_len is None:
            self.waveform_len = x.size(2)
        _, s, _, _, _ = self.model.encode(x)
        s = normalize(s, self.old_range, self.new_range).float()
        return s
    
    @torch.no_grad()
    def decode(self, s : Tensor) -> Tensor:
        s = normalize(s, self.new_range, self.old_range).round().clamp(*self.old_range).long()
        zq = self.s_to_zq(s)
        x = self.model.decode(zq)
        # pad to original length
        x = torch.nn.functional.pad(x, (0, self.waveform_len - x.size(2)), mode='constant', value=0)
        return x