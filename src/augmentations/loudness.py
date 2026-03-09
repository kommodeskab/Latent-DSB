from src.augmentations.base import BaseAugmentation
from src import AudioSample
import torch


class LoudnessAugmentation(BaseAugmentation):
    def __init__(
        self,
        loudness_db: float,
    ):
        self.loudness_db = loudness_db
        self.epsilon = 1e-6

    def _rms_dbfs(self, waveform: torch.Tensor) -> torch.Tensor:
        # dbfs = "decibels relative to full scale"
        rms = waveform.pow(2).mean().sqrt().clamp(min=self.epsilon)
        return 20.0 * torch.log10(rms + self.epsilon)

    def __call__(self, sample: AudioSample) -> AudioSample:
        waveform = sample["waveform"]

        current_dbfs = self._rms_dbfs(waveform)
        gain_db = self.loudness_db - current_dbfs
        gain = 10.0 ** (gain_db / 20.0)
        augmented_waveform = waveform * gain

        return AudioSample(
            waveform=augmented_waveform,
            sample_rate=sample["sample_rate"],
        )


if __name__ == "__main__":
    # Example usage
    sample = AudioSample(
        waveform=torch.randn(1, 16000),  # 1 second of audio at 16kHz
        sample_rate=16000,
    )

    print(sample["waveform"].min(), sample["waveform"].max())

    augmentation = LoudnessAugmentation(loudness_db=0.0)
    augmented_sample = augmentation(sample)

    print(augmented_sample["waveform"].min(), augmented_sample["waveform"].max())
