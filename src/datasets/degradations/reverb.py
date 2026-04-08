from src import AudioSample
from src.datasets.degradations import BaseDegradation
from src.datasets.audio import AudioDataset
import torch
from torch import Tensor
from torchaudio.functional import fftconvolve
from src.datasets import RIR


class Reverb(BaseDegradation):
    """
    Helper class that applies a RIR to an audio tensor.
    The RIR is sampled from the given RIR dataset.
    """

    def __init__(
        self, RIR_dataset: AudioDataset, prob: float = 1.0, deterministic: bool = False, rir_threshold: float = -20.0
    ):
        super().__init__(prob=prob, deterministic=deterministic)
        self.RIR_dataset = RIR_dataset
        self.rir_threshold = rir_threshold

    def fun(self, audio: Tensor) -> Tensor:
        RIR: AudioSample = self.RIR_dataset.sample()

        # Filter the start of the rir at the rir threshold.
        energy = RIR["waveform"] ** 2

        # linear scale threshold
        threshold = energy.max() * (10 ** (self.rir_threshold / 10.0))

        # find first sample over threshold. [0] gives first, [-1] gives sample dim index
        delay_idx = (energy >= threshold).nonzero()[0][-1]
        rir = RIR["waveform"][..., delay_idx:]
        # normalize to original loudness - ensure no energy is added by the rir.
        audio_len = audio.shape[1]
        reverbed_audio = fftconvolve(audio, rir, mode="full")[..., :audio_len]
        power_reverb = torch.mean(reverbed_audio**2)
        power_original = torch.mean(audio**2)

        return reverbed_audio * torch.sqrt(power_original / power_reverb)


if __name__ == "__main__":
    rir_dataset = RIR(split="train")
    audio = torch.rand((1, 48000))
    test_reverb = Reverb(rir_dataset, prob=1.0)
    print(audio.shape)
    print(test_reverb(audio).shape)
