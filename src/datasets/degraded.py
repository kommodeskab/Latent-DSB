from src.datasets.audio import AudioDataset, BaseDataset
from src import DegradedAudioSample
from src.datasets.degradations import BaseDegradation


class DegradedDataset(BaseDataset):
    def __init__(
        self,
        clean_dataset: AudioDataset,
        degradations: list[BaseDegradation],
    ):
        super().__init__()
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
