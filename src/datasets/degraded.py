from src.datasets.audio import AudioDataset, BaseDataset
from src import DegradedAudioSample
from src.datasets.degradations import BaseDegradation
from src.utils import get_context
import torch


class DegradedDataset(BaseDataset):
    def __init__(
        self,
        dataset: AudioDataset,
        degradations: list[BaseDegradation],
        probs: list[float] | float = 1.0,
        deterministic: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.degradations = degradations
        self.probs = probs if isinstance(probs, list) else [probs] * len(degradations)
        self.deterministic = deterministic

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DegradedAudioSample:
        clean = self.dataset[idx]
        clean_waveform = clean["waveform"]
        noisy_waveform = clean_waveform.clone()

        for degradation, prob in zip(self.degradations, self.probs):
            with get_context(seed=idx, deterministic=self.deterministic):
                if torch.rand(1).item() < prob:
                    noisy_waveform = degradation(noisy_waveform)

        return DegradedAudioSample(
            original_waveform=clean_waveform,
            degraded_waveform=noisy_waveform,
            sample_rate=clean["sample_rate"],
        )
