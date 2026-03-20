from torch import Tensor
import torch
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from transformers import SpeechT5HifiGan
from functools import partial


class BaseEncoderDecoder:
    def encode(self, x: Tensor) -> Tensor: ...
    def decode(self, h: Tensor) -> Tensor: ...
    def to(self, device: torch.device): ...


class IdentityEncoderDecoder(BaseEncoderDecoder):
    def __init__(self, scaling_factor: float = 1.0):
        self.scaling_factor = scaling_factor

    def encode(self, x: Tensor) -> Tensor:
        return x * self.scaling_factor

    def decode(self, h: Tensor) -> Tensor:
        return h / self.scaling_factor


class STFTEncoderDecoder(BaseEncoderDecoder):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        alpha: float = 1 / 4,
        beta: float = 4,
        image_like: bool = False,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.alpha = alpha
        self.beta = beta
        self.image_like = image_like

    def encode(self, audio: Tensor) -> Tensor:
        if not hasattr(self, "original_length"):
            self.original_length = audio.shape[-1]

        audio = torch.cat(
            (audio, torch.zeros(audio.shape[0], audio.shape[1], self.n_fft - 1, device=audio.device)), dim=-1
        )
        audio = audio.squeeze(1)
        window = torch.hamming_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            window=window,
            return_complex=True,
        )

        stft = self.beta * stft.abs() ** self.alpha * torch.exp(1j * stft.angle())
        real, imag = stft.real, stft.imag
        # real.shape = imag.shape = (batch_size, n_freq_bins, n_frames)
    
        if self.image_like:
            # make two channels, i.e. shape (batch_size, 2, n_freq_bins, n_frames)
            out = torch.stack([real, imag], dim=1)
        else:
            # stack frequency bins, i.e. shape (batch_size, 2*n_freq_bins, n_frames)
            out = torch.cat([real, imag], dim=1)

        return out

    def decode(self, encoded: Tensor) -> Tensor:
        if self.image_like:
            real, imag = encoded[:, 0], encoded[:, 1]
        else:
            n_freq_bins = encoded.shape[1] // 2
            real, imag = encoded[:, :n_freq_bins], encoded[:, n_freq_bins:]

        stft = (real + 1j * imag) / self.beta
        stft = stft.abs() ** (1 / self.alpha) * torch.exp(1j * stft.angle())

        window = torch.hamming_window(self.win_length, device=stft.device)
        audio = torch.istft(
            stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=False, window=window
        )
        audio = torch.nn.functional.pad(audio, (0, self.original_length - audio.shape[-1]), mode="constant", value=0)
        return audio.unsqueeze(1)


class PolarSTFTEncoderDecoder(BaseEncoderDecoder):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

    def encode(self, audio: Tensor) -> Tensor:
        if not hasattr(self, "original_length"):
            self.original_length = audio.shape[-1]

        audio = torch.cat(
            (audio, torch.zeros(audio.shape[0], audio.shape[1], self.n_fft - 1, device=audio.device)), dim=-1
        )
        audio = audio.squeeze(1)
        window = torch.hamming_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            window=window,
            return_complex=True,
        )

        log_amplitude = (1e-6 + stft.abs()).log()
        angle = stft.angle()
        cos = 3 * angle.cos()  # multiply by 3 to bring on (approx) same scale as log_amplitude
        sin = 3 * angle.sin()
        out = torch.stack([log_amplitude, cos, sin], dim=1)
        return out

    def decode(self, encoded: Tensor) -> Tensor:
        log_amplitude, cos, sin = encoded[:, 0], encoded[:, 1], encoded[:, 2]
        cos, sin = cos / 3, sin / 3  # undo scaling
        sin = sin.clamp(-1.0, 1.0)
        cos = cos.clamp(-1.0, 1.0)
        angle = torch.atan2(sin, cos)
        magnitude = log_amplitude.exp()
        stft = magnitude * torch.exp(1j * angle)

        window = torch.hamming_window(self.win_length, device=stft.device)
        audio = torch.istft(
            stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=False, window=window
        )
        audio = torch.nn.functional.pad(audio, (0, self.original_length - audio.shape[-1]), mode="constant", value=0)
        return audio.unsqueeze(1)


class HifiGan(BaseEncoderDecoder):
    def __init__(self):
        super().__init__()
        self.vocoder: SpeechT5HifiGan = SpeechT5HifiGan.from_pretrained("cvssp/audioldm2", subfolder="vocoder")
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
            compression=True,
        )

        self.original_len = None

    def to(self, device: torch.device):
        self.vocoder = self.vocoder.to(device)

    def encode(self, x: Tensor) -> Tensor:
        if self.original_len is None:
            self.original_len = x.shape[2]

        x_mel = [self.to_mel(audio=audio.squeeze())[0] for audio in x]
        x_mel = torch.stack(x_mel, dim=0)
        x_mel = x_mel.unsqueeze(1)  # add channel dimension
        return x_mel

    def decode(self, x: Tensor) -> Tensor:
        x = x.squeeze(1)  # remove channel dimension
        x = x.permute(0, 2, 1)
        decoded: Tensor = self.vocoder(x)
        decoded = decoded.unsqueeze(1)
        if self.original_len is not None:
            decoded = torch.nn.functional.pad(
                decoded, (0, self.original_len - decoded.size(2)), mode="constant", value=0
            )
        return decoded
