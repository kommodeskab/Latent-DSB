from src.augmentations.base import BaseAugmentation
from src import AudioSample
import torch


class NormalizeAugmentation(BaseAugmentation):
    """
    Normalizes an audio sample to have zero mean and unit variance/standard deviation.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, sample: AudioSample) -> AudioSample:
        waveform = sample["waveform"]
        normalized_waveform = (waveform - waveform.mean()) / (waveform.std() + self.eps)

        return AudioSample(
            waveform=normalized_waveform,
            sample_rate=sample["sample_rate"],
        )


if __name__ == "__main__":
    # Example usage
    sample = AudioSample(
        waveform=torch.randn(1, 16000) * 5 + 3,  # 1 second of audio at 16kHz
        sample_rate=16000,
    )

    augmentation = NormalizeAugmentation()
    augmented_sample = augmentation(sample)

    print("Mean:", augmented_sample["waveform"].mean().item())
    print("Std:", augmented_sample["waveform"].std().item())
