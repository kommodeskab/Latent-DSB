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
        self,
        RIR_dataset: AudioDataset,
        prob: float = 0.0,
        deterministic: bool = False,
    ):

        super().__init__(prob=prob,deterministic=deterministic)
        self.RIR_dataset = RIR_dataset
        self.deterministic = deterministic

    def fun(self, audio: Tensor) -> Tensor:
        RIR: AudioSample = self.RIR_dataset.sample()
        return fftconvolve(audio,RIR["waveform"],mode="same")


if __name__ == "__main__":
    rir_dataset = RIR(split="train")
    audio = torch.rand((1,48000))
    test_reverb = Reverb(rir_dataset,prob=1.0)
    print(audio.shape)
    print(test_reverb(audio).shape)