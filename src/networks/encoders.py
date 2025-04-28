from diffusers import VQModel, AutoencoderKL
from torch import Tensor
from torch.nn import Module
from transformers import MimiModel, AutoFeatureExtractor
from torch import Tensor
import torch
import dac
from diffusers import AutoencoderOobleck
from functools import partial
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from speechbrain.inference.vocoders import HIFIGAN
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
        return encoder
    
class AudioLDMEncoder:
    def __new__(cls):
        encoder = PretrainedVAE("cvssp/audioldm2", subfolder="vae", revision=None, variant=None)
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
    def __init__(
        self, 
        n_fft : int = 510, 
        hop_length : int = 258, 
        ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.original_length = None
        
    def encode(self, audio : Tensor) -> Tensor:
        if self.original_length is None:
            self.original_length = audio.size(2)
            self.window = torch.hamming_window(self.n_fft, device=audio.device)
            
        audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        spec = torch.view_as_real(spec).permute(0, 3, 1, 2)
        return spec
    
    def decode(self, spec : Tensor) -> Tensor:
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
        for param in self.model.parameters():
            param.requires_grad = False
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
    
class StableAudioEncoder(Module):
    def __init__(self, chunk_size : int = 32):
        super().__init__()
        self.autoencoder : AutoencoderOobleck = AutoencoderOobleck.from_pretrained(
            'stabilityai/stable-audio-open-1.0', 
            subfolder='vae', 
            variant=None,
            token='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',
            torch_dtype=torch.float16,
        )
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        self.sample_rate = 44100
        # chunk size is used for chunked encoding and decoding to avoid oom errors
        # batch is chunked into chunks of this size
        self.chunk_size = chunk_size
        self.waveform_len = None
        
    def _encode(self, x : Tensor) -> Tensor:
        # helper function to encode a chunk of audio
        if self.waveform_len is None:
            self.waveform_len = x.size(2)
        if x.size(1) == 1:
            x = x.repeat(1, 2, 1)
        return self.autoencoder.encode(x).latent_dist.sample()
    
    def _decode(self, h : Tensor) -> Tensor:
        # helper function to decode a chunk of audio
        x = self.autoencoder.decode(h).sample
        x = x[:, :1, :]
        x = torch.nn.functional.pad(x, (0, self.waveform_len - x.size(2)), mode='constant', value=0)
        return x
        
    def encode(self, x : Tensor) -> Tensor:
        x_chunked = x.split(self.chunk_size, dim=0)
        h = []
        for chunk in x_chunked:
            h_chunk = self._encode(chunk)
            h.append(h_chunk)
        h = torch.cat(h, dim=0)
        return h
    
    def decode(self, h : Tensor) -> Tensor:
        h_chunked = h.split(self.chunk_size, dim=0)
        x = []
        for chunk in h_chunked:
            x_chunk = self._decode(chunk)
            x.append(x_chunk)
        x = torch.cat(x, dim=0)
        return x
    
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

class __HifiGan(Module):
    def __init__(self):
        super().__init__()
        self.hifigan : HIFIGAN = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="/work3/s214630/data/pretrained_models/tts-hifigan-libritts-16kHz")
        for param in self.hifigan.parameters():
            param.requires_grad = False
        self.hifigan.eval()
        self.to_mel = partial(
            mel_spectogram,
            sample_rate=16000,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            n_fft=1024,
            f_min=0.0,
            f_max=8000.0,
            power=1,
            normalized=False,
            min_max_energy_norm=True,
            norm="slaney",
            mel_scale="slaney",
            compression=True
        )
        self.original_len = None
    
    def encode(self, x : Tensor) -> Tensor:
        # x is audio with shape (batch_size, 1, seq_len)
        # make x into list of tensors
        if self.original_len is None:
            self.original_len = x.shape[2]
        x_list = x.split(1, dim=0)
        x_mel = [self.to_mel(audio = x.squeeze())[0] for x in x_list]
        x_mel = torch.stack(x_mel, dim=0)
        # standardize
        x_mel = x_mel + 4
        return x_mel
    
    def decode(self, h : Tensor) -> Tensor:
        h = h - 4
        decoded = self.hifigan.decode_batch(h)
        if self.original_len is not None:
            decoded = torch.nn.functional.pad(decoded, (0, self.original_len - decoded.size(2)), mode='constant', value=0)
        return decoded
    
    def to(self, device : torch.device):
        self.hifigan.device = device
        return super().to(device)
    
class OpenSoundEncoder(Module):
    def __init__(self):
        super().__init__()
        print("Loading OpenSound VAE...")
        from stable_audio_tools import create_model_from_config_path
        vae = create_model_from_config_path('open-sound-vae/config.json')
        ckpt = torch.load("open-sound-vae/1500k.ckpt", map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"]
        state_dict = {k[len('autoencoder.'):]: v for k, v in state_dict.items() if k.startswith('autoencoder.')}
        vae.load_state_dict(state_dict, strict=True)
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        self.vae = vae
        
    def encode(self, x : Tensor) -> Tensor:
        return self.vae.encode(x)
    
    def decode(self, x : Tensor) -> Tensor:
        return self.vae.decode(x)