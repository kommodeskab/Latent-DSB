from torch import Tensor
from torch.nn import Module
import torch


class BaseEncoderDecoder:
    def encode(self, x: Tensor) -> Tensor: ...
    def decode(self, h: Tensor) -> Tensor: ...


class IdentityEncoderDecoder(Module):
    def __init__(self, scaling_factor: float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def encode(self, x: Tensor) -> Tensor:
        return x * self.scaling_factor

    def decode(self, h: Tensor) -> Tensor:
        return h / self.scaling_factor


class STFTEncoderDecoder:
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        alpha: float = 1 / 4,
        beta: float = 4,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.alpha = alpha
        self.beta = beta

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

        out = torch.stack([real, imag], dim=1)

        return out

    def decode(self, encoded: Tensor) -> Tensor:
        real, imag = encoded[:, 0], encoded[:, 1]

        stft = (real + 1j * imag) / self.beta
        stft = stft.abs() ** (1 / self.alpha) * torch.exp(1j * stft.angle())

        window = torch.hamming_window(self.win_length, device=stft.device)
        audio = torch.istft(
            stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=False, window=window
        )
        audio = torch.nn.functional.pad(audio, (0, self.original_length - audio.shape[-1]), mode="constant", value=0)
        return audio.unsqueeze(1)


class PolarSTFTEncoderDecoder:
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
