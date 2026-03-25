from src.datasets.audio import BaseDataset
from src.datasets.degraded import DegradedDataset
from src import UnpairedAudioBatch
from src.utils import get_context


class UnpairedAudioDataset(BaseDataset):
    """
    Given some degraded dataset, return a random "coupling", i.e. two random samples from the dataset,
    where one (x0) is the original and the other (x1) is the degraded version.
    The dataset also returns the clean version of x1 (x1_clean). This is useful for calculating some metrics.
    """

    def __init__(self, dataset: DegradedDataset, deterministic: bool = False, paired: bool = False):
        super().__init__()
        self.dataset = dataset
        self.deterministic = deterministic
        self.paired = paired

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> UnpairedAudioBatch:
        x1 = self.dataset[idx]
        x1_waveform = x1["degraded_waveform"]

        if self.paired:
            x0_waveform = x1["original_waveform"]
        else:
            with get_context(idx, self.deterministic):
                x0 = self.dataset.sample()
            x0_waveform = x0["original_waveform"]

        return UnpairedAudioBatch(
            x0=x0_waveform,
            x1=x1_waveform,
            x1_clean=x1["original_waveform"],
            sample_rate=x1["sample_rate"],
        )
