from src.datasets.audio import AudioDataset
from src import DegradedAudioSample
from src.datasets.degradations import BaseDegradation, AddNoise

class DegradedDataset(AudioDataset):
    """
    Given a dataset, this dataset applies some kind of degradation to the audio samples.
    __getitem__ then returns a "DegradedAudioSample" which has both the original and degraded audio, as well as the sample rate.
    """

    def __getitem__(self, idx: int) -> DegradedAudioSample: ...


class NoisyDegradedDataset(DegradedDataset):
    def __init__(
        self,
        clean_dataset: AudioDataset,
        noise_dataset: AudioDataset,
        min_snr: float,
        max_snr: float,
        deterministic: bool = False,
    ):
        self.clean_dataset = clean_dataset
        self.add_noise = AddNoise(
            noise_dataset = noise_dataset,
            min_snr = min_snr,
            max_snr = max_snr,
            deterministic = deterministic,
        )

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx: int) -> DegradedAudioSample:
        clean = self.clean_dataset[idx]
        noisy_waveform = self.add_noise(clean["waveform"], seed=idx)

        return DegradedAudioSample(
            original_waveform=clean["waveform"],
            degraded_waveform=noisy_waveform,
            sample_rate=clean["sample_rate"],
        )


# TODO: implement a "ReverberantDegradedDataset" that adds reverb to the clean samples,
# using some kind of impulse response dataset
# good idea to find or upload a reverb dataset on HugginFace or similar

class VeryDegradedDataset(DegradedDataset):
    def __init__(
        self,
        clean_dataset: AudioDataset,
        degradations: list[BaseDegradation],
    ):
        self.clean_dataset = clean_dataset
        self.degradations = degradations
        
    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self, idx: int) -> DegradedAudioSample:
        clean = self.clean_dataset[idx]
        clean_waveform = clean["waveform"]
        noisy_waveform = clean_waveform.clone()
        
        for degradation in self.degradations:
            noisy_waveform = degradation(noisy_waveform, seed=idx)

        return DegradedAudioSample(
            original_waveform=clean_waveform,
            degraded_waveform=noisy_waveform,
            sample_rate=clean["sample_rate"],
        )