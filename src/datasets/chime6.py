import glob
import os
from typing import Literal

from huggingface_hub import snapshot_download

from src import AudioPathSample
from src.datasets import AudioPathDataset


class CHiME6(AudioPathDataset):
    """
    CHiME-6 audio dataset from HuggingFace (https://huggingface.co/datasets/thevoicecompany/chime).

    Downloads the dataset from HuggingFace Hub to `self.data_path` and indexes audio files.
    Applies a 90/10 train/val split over all available audio files.
    Each item returned by `__getitem__` is a dictionary containing the audio file path key 'audio_path'.
    """

    def __init__(self, split: Literal["train", "val"] = "train"):
        super().__init__()
        self.split = split

        # Download dataset repository from HuggingFace to self.data_path
        local_dir = snapshot_download(
            repo_id="thevoicecompany/chime",
            repo_type="dataset",
            cache_dir=self.data_path,
        )

        # Collect all audio .wav file paths deterministically
        all_audio_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.wav"), recursive=True))

        # Perform a 90/10 split between train and val
        split_idx = int(0.9 * len(all_audio_files))

        if split == "train":
            self.audio_files = all_audio_files[:split_idx]
        elif split == "val":
            self.audio_files = all_audio_files[split_idx:]
        else:
            raise ValueError(f"Unsupported split: {split}. Expected 'train' or 'val'.")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> AudioPathSample:
        """
        Returns a dictionary containing the audio file path under key 'audio_path'.

        Args:
            index (int): Index of the sample.

        Returns:
            Dict[str, str]: Dictionary with key 'audio_path'.
        """
        return {"audio_path": self.audio_files[index]}


if __name__ == "__main__":
    from src.datasets import SegmentedAudioDataset

    chime_dataset = CHiME6(split="train")
    subset_size_sec = 10.0
    segmented_dataset = SegmentedAudioDataset(chime_dataset, subset_size=subset_size_sec)

    print(f"CHiME6 dataset length: {len(chime_dataset)}")
    print(f"Segmented CHiME6 dataset length: {len(segmented_dataset)}")

    sample = segmented_dataset[0]
    waveform = sample["waveform"]
    sample_rate = sample["sample_rate"]
    print(f"Sample waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")
    expected_samples = int(subset_size_sec * sample_rate)
    print(f"Expected number of samples: {expected_samples}")
