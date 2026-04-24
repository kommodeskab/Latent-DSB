from src.datasets.audio import BaseDataset
from src.datasets.degraded import DegradedDataset, AudioDataset
from src import UnpairedAudioBatch
from src.utils import get_context
from typing import Optional, Union


class UnpairedAudioDataset(BaseDataset):
    """

    This dataset class is designed to handle various settings in a unpaired training scenario.

    If only a degraded dataset is provided:
    - If paired is False, then the clean and degraded waveforms are sampled randomly from the degraded_dataset.
    - If paired is True, then the clean and degraded waveforms are sampled from the same index of the degraded_dataset.

    If a separate clean dataset is provided:
    - Paired can only be False. In this case, the degraded waveform is sampled from the degraded_dataset and the clean waveform is sampled randomly from the clean_dataset.

    The keys for the degraded and clean waveforms can be specified separately, allowing for flexibility in dataset structure.
    Additionally, an optional key for a clean version of the degraded waveform can be provided which is needed for some metrics.

    Args:
        degraded_dataset (Union[DegradedDataset, AudioDataset]): The dataset containing the degraded audio samples.
        x0_key (str): The key in the degraded_dataset that corresponds to the clean waveform.
        x1_key (str): The key in the degraded_dataset (or in the clean_dataset, if provided) that corresponds to the degraded waveform.
        x1_clean_key (Optional[str]): An optional key in the degraded_dataset that corresponds to a clean version of the degraded waveform, which is needed for some metrics.
        clean_dataset (Optional[Union[DegradedDataset, AudioDataset]]): An optional dataset containing clean audio samples. If not provided, the clean samples will be sampled from the degraded_dataset.
        deterministic (bool): Whether to use deterministic sampling. If True, the same samples will be returned for the same index across epochs. If False, the samples will be randomly sampled each time.
        paired (bool): Whether to sample clean and degraded waveforms from the same index of the degraded_dataset. If True, clean and degraded waveforms will be sampled from the same index of the degraded_dataset. If False, clean and degraded waveforms will be sampled randomly from the degraded_dataset (if clean_dataset is not provided) or from the degraded_dataset and clean_dataset respectively.

    """

    def __init__(
        self,
        degraded_dataset: Union[DegradedDataset, AudioDataset],
        x0_key: str,
        x1_key: str,
        x1_clean_key: Optional[str] = None,
        clean_dataset: Optional[Union[DegradedDataset, AudioDataset]] = None,
        deterministic: bool = False,
        paired: bool = False,
    ):
        super().__init__()
        if paired:
            assert (
                clean_dataset is None
            ), "If paired is True, then clean_dataset should not be provided. The clean and degraded waveforms will be sampled from the same index of the degraded_dataset."

        self.degraded_dataset = degraded_dataset
        self.clean_dataset = clean_dataset if clean_dataset is not None else degraded_dataset
        self.deterministic = deterministic
        self.paired = paired
        self.x0_key = x0_key
        self.x1_key = x1_key
        self.x1_clean_key = x1_clean_key

    def __len__(self) -> int:
        return len(self.degraded_dataset)

    def __getitem__(self, idx: int) -> UnpairedAudioBatch:
        with get_context(idx, self.deterministic):
            degraded = self.degraded_dataset[idx]
            # if "paired" is true, that means we are training on paired data from the same degraded dataset
            # otherwise, we sample from the clean dataset.
            # if no clean dataset is provided during initialization, then the clean dataset is the same as the degraded dataset, but they key might be different
            clean = degraded if self.paired else self.clean_dataset.sample()

            degraded_waveform = degraded[self.x1_key]
            clean_waveform = clean[self.x0_key]

            assert (
                clean["sample_rate"] == degraded["sample_rate"]
            ), "Sample rates of clean and degraded datasets must match."

        return UnpairedAudioBatch(
            x0=clean_waveform,
            x1=degraded_waveform,
            x1_clean=degraded.get(
                self.x1_clean_key, -1
            ),  # if x1_clean_key is not provided, we set it to -1 (like None but dataloader doesn't like None)
            sample_rate=degraded["sample_rate"],
        )
