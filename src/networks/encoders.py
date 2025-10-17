from torch import Tensor
from torch.nn import Module
from torch import Tensor
import torch
import dac
from functools import partial
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from transformers import SpeechT5HifiGan
from diffusers import AutoencoderKL

class BaseEncoderDecoder(Module):
    def encode(self, x : Tensor) -> Tensor: ...    
    def decode(self, h : Tensor) -> Tensor: ...
    
class IdentityEncoderDecoder(Module):
    def __init__(self, scaling_factor : float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
    def encode(self, x : Tensor) -> Tensor: return x * self.scaling_factor
    def decode(self, h : Tensor) -> Tensor: return h / self.scaling_factor
    
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
        return encoder

def normalize(x : Tensor, old_range : tuple, new_range : tuple) -> Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

class OpenSoundEncoder(Module):
    def __init__(self):
        super().__init__()
        print("Loading OpenSound VAE...")
        from stable_audio_tools import create_model_from_config_path
        vae = create_model_from_config_path('open-sound-vae/config.json')
        ckpt = torch.load("open-sound-vae/1500k.ckpt", map_location="cpu", weights_only=True)
        state_dict : dict[str, Tensor] = ckpt["state_dict"]
        state_dict = {k[len('autoencoder.'):]: v for k, v in state_dict.items() if k.startswith('autoencoder.')}
        vae.load_state_dict(state_dict, strict=True)
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        self.vae = vae
        
    def encode(self, x : Tensor) -> Tensor:
        self.original_length = x.shape[-1]
        return self.vae.encode(x).unsqueeze(1)
    
    def decode(self, x : Tensor) -> Tensor:
        x = x.squeeze(1)
        audio = self.vae.decode(x)
        audio = torch.nn.functional.pad(audio, (0, self.original_length - audio.shape[-1]), mode='constant', value=0)
        return audio

class STFTEncoderDecoder(Module):
    def __init__(
        self,
        n_fft : int,
        hop_length : int,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        
    def encode(self, audio : Tensor) -> Tensor:
        if not hasattr(self, 'original_length'):
            self.original_length = audio.shape[-1]
        
        audio = torch.cat((audio, torch.zeros(audio.shape[0], audio.shape[1], self.n_fft - 1, device=audio.device)), dim=-1)
        audio = audio.squeeze(1)
        window = torch.hamming_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            center=False,
            window=window,
            return_complex=True
            )

        out = torch.stack([stft.real, stft.imag], dim=1)

        return out
    
    def decode(self, encoded : Tensor) -> Tensor:
        real, imag = encoded[:, 0], encoded[:, 1]
        stft = real + 1j * imag

        window = torch.hamming_window(self.win_length, device=stft.device)
        audio = torch.istft(
            stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            center=False,
            window=window
        )
        audio = torch.nn.functional.pad(audio, (0, self.original_length - audio.shape[-1]), mode='constant', value=0)
        return audio.unsqueeze(1)
    
class PolarSTFTEncoderDecoder(Module):
    def __init__(
        self,
        n_fft : int,
        hop_length : int,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        
    def encode(self, audio : Tensor) -> Tensor:
        if not hasattr(self, 'original_length'):
            self.original_length = audio.shape[-1]
        
        audio = torch.cat((audio, torch.zeros(audio.shape[0], audio.shape[1], self.n_fft - 1, device=audio.device)), dim=-1)
        audio = audio.squeeze(1)
        window = torch.hamming_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            center=False,
            window=window,
            return_complex=True
            )
        
        log_amplitude = (1e-6 + stft.abs()).log()
        angle = stft.angle()
        cos = 3 * angle.cos() # multiply by 3 to bring on (approx) same scale as log_amplitude
        sin = 3 * angle.sin()
        out = torch.stack([log_amplitude, cos, sin], dim=1)
        return out
    
    def decode(self, encoded : Tensor) -> Tensor:
        log_amplitude, cos, sin = encoded[:, 0], encoded[:, 1], encoded[:, 2]
        cos, sin = cos / 3, sin / 3 # undo scaling
        sin = sin.clamp(-1.0, 1.0)
        cos = cos.clamp(-1.0, 1.0)
        angle = torch.atan2(sin, cos)
        magnitude = log_amplitude.exp()
        stft = magnitude * torch.exp(1j * angle)
        
        window = torch.hamming_window(self.win_length, device=stft.device)
        audio = torch.istft(
            stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            center=False,
            window=window
        )
        audio = torch.nn.functional.pad(audio, (0, self.original_length - audio.shape[-1]), mode='constant', value=0)
        return audio.unsqueeze(1)
    
class HifiGan(Module):
    def __init__(self):
        super().__init__()
        self.vocoder : SpeechT5HifiGan = SpeechT5HifiGan.from_pretrained("cvssp/audioldm2", subfolder="vocoder")
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.vocoder.eval()
        
        self.to_mel = partial(
            mel_spectogram,
            sample_rate=16000,
            hop_length=160,
            win_length=1024,
            n_mels=64,
            n_fft=1024,
            f_min=0.0,
            f_max=8000.0,
            power=1,
            normalized=False,
            min_max_energy_norm=False,
            norm="slaney",
            mel_scale="slaney",
            compression=True
        )
        
        self.original_len = None
        self.off_set = 4 # approximately normalize between -4 and 4
    
    def encode(self, x : Tensor) -> Tensor:
        if self.original_len is None:
            self.original_len = x.shape[2]
            
        x_mel = [self.to_mel(audio = audio.squeeze())[0] for audio in x]
        x_mel = torch.stack(x_mel, dim=0)
        x_mel = x_mel + self.off_set
        return x_mel
    
    def decode(self, x : Tensor) -> Tensor:
        x = x - self.off_set # undo normalization
        x = x.permute(0, 2, 1)
        decoded : Tensor = self.vocoder(x)
        decoded = decoded.unsqueeze(1)
        if self.original_len is not None:
            decoded = torch.nn.functional.pad(decoded, (0, self.original_len - decoded.size(2)), mode='constant', value=0)
        return decoded