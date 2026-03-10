from src.datasets.audio import BaseDataset
from src.datasets.degraded import DegradedDataset
from src import UnpairedAudioBatch
from src.utils import temporary_seed
from contextlib import nullcontext


class UnpairedAudioDataset(BaseDataset):
    """
    Given some degraded dataset, return a random "coupling", i.e. two random samples from the dataset,
    where one (x0) is the original and the other (x1) is the degraded version.
    The dataset also returns the clean version of x1 (x1_clean). This is useful for calculating some metrics.
    """

    def __init__(self, dataset: DegradedDataset, deterministic: bool = False):
        super().__init__()
        self.dataset = dataset
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> UnpairedAudioBatch:
        x1 = self.dataset[idx]
        context = temporary_seed(idx) if self.deterministic else nullcontext()
        with context:  # ensure that the random coupling is deterministic if desired
            x0 = self.dataset.sample()

        # x1 is a sample from the same dataset. yes, there is a slight possibility that
        # x1 is the same sample as x0, but that's fine since
        # the dataset is large and this is just a random coupling.

        sr0 = x0["sample_rate"]

        return UnpairedAudioBatch(
            x0=x0["original_waveform"],
            x1=x1["degraded_waveform"],
            x1_clean=x1["original_waveform"],
            sample_rate=sr0,
        )
