from torch import Tensor
from torch.nn import Module
from torch import Tensor
import torch
import dac
from functools import partial
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from transformers import SpeechT5HifiGan

class BaseEncoderDecoder(Module):
    def encode(self, x : Tensor) -> Tensor: ...    
    def decode(self, h : Tensor) -> Tensor: ...
    
class IdentityEncoderDecoder(Module):
    def __init__(self, scaling_factor : float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
    def encode(self, x : Tensor) -> Tensor: return x * self.scaling_factor
    def decode(self, h : Tensor) -> Tensor: return h / self.scaling_factor

def normalize(x : Tensor, old_range : tuple, new_range : tuple) -> Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

class STFTEncoderDecoder(Module):
    def __init__(
        self, 
        n_fft : int = 510, 
        hop_length : int = 258, 
        one_dimensional : bool = False
        ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.original_length = None
        self.one_dimensional = one_dimensional
        
    def encode(self, audio : Tensor) -> Tensor:
        if self.original_length is None:
            self.original_length = audio.size(2)
            self.window = torch.hamming_window(self.n_fft, device=audio.device)

        audio = torch.cat((audio, torch.zeros(audio.shape[0], audio.shape[1], self.n_fft - 1, device=audio.device)), dim=-1)
        audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        spec = torch.view_as_real(spec).permute(0, 3, 1, 2)
        if self.one_dimensional:
            B, C, F, T = spec.shape
            spec = spec.reshape(B, C * F, T)
        return spec
    
    def decode(self, spec : Tensor) -> Tensor:
        if self.one_dimensional:
            B, C, T = spec.shape
            F = C // 2
            spec = spec.reshape(B, 2, F, T)
        spec = spec.permute(0, 2, 3, 1).contiguous()
        spec = torch.view_as_complex(spec)
        
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=False,
            center=False,
            window=self.window,
        )
        audio = audio.unsqueeze(1)
        audio = torch.nn.functional.pad(audio, (0, self.original_length - audio.size(2)), mode='constant', value=0)
        return audio
    
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