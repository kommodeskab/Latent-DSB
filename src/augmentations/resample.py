from src.augmentations.base import BaseAugmentation
from torchaudio.functional import resample
from src import AudioSample


class ResampleAugmentation(BaseAugmentation):
    def __init__(self, new_sample_rate: int):
        self.new_sample_rate = new_sample_rate

    def __call__(self, sample: AudioSample) -> AudioSample:
        if sample["sample_rate"] == self.new_sample_rate:
            return sample

        resampled_waveform = resample(
            sample["waveform"],
            orig_freq=sample["sample_rate"],
            new_freq=self.new_sample_rate,
        )

        return AudioSample(
            waveform=resampled_waveform,
            sample_rate=self.new_sample_rate,
        )
