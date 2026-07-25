import os
from typing import List, Tuple, Union

import soundfile as sf
import torch

from src import AudioSample
from src.datasets import BaseDataset
from src.datasets import AudioPathDataset


class SegmentedAudioDataset(BaseDataset):
    """
    A helper dataset that takes an underlying dataset of audio file paths (such as CHiME6)
    and splits each audio file into smaller, non-overlapping segments of size `subset_size` (in seconds).

    For example, if an audio file in the underlying dataset is 100 seconds long and `subset_size`
    is 10 seconds, this dataset creates 10 distinct items of 10 seconds each.
    The resulting dataset has more entries, each returning a segment waveform of length `subset_size` seconds.

    Args:
        dataset: Underlying dataset where each item is a dict containing key 'audio_path' (or 'path').
        subset_size (float | int): Target segment duration in seconds (e.g. 10.0 for 10 seconds).
        drop_last (bool): Whether to drop trailing segments that are shorter than `subset_size`.
            Default is False.
    """

    def __init__(
        self,
        dataset: AudioPathDataset,
        subset_size: Union[float, int],
        drop_last: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.subset_size = float(subset_size)
        self.drop_last = drop_last

        # Index all segments across all items in the underlying dataset
        # List of tuples: (path, start_frame, target_frames, sample_rate)
        self.segments: List[Tuple[str, int, int, int]] = []

        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            path = item.get("audio_path")

            with sf.SoundFile(path) as f:
                sample_rate = f.samplerate
                total_frames = len(f)

            target_frames = int(self.subset_size * sample_rate)

            if target_frames <= 0:
                raise ValueError(f"subset_size in seconds must be positive, got {subset_size}")

            if total_frames < target_frames:
                if not drop_last:
                    self.segments.append((path, 0, target_frames, sample_rate))
            else:
                if drop_last:
                    num_segments = total_frames // target_frames
                else:
                    num_segments = (total_frames + target_frames - 1) // target_frames

                for seg_idx in range(num_segments):
                    start_frame = seg_idx * target_frames
                    self.segments.append((path, start_frame, target_frames, sample_rate))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> AudioSample:
        """
        Loads the specific audio segment from disk.

        Args:
            index (int): Index of the segment.

        Returns:
            AudioSample: Dictionary containing 'waveform' (Tensor) and 'sample_rate' (int).
        """
        path, start_frame, target_frames, sample_rate = self.segments[index]

        with sf.SoundFile(path) as f:
            f.seek(start_frame)
            audio_data = f.read(target_frames, dtype="float32", always_2d=True)

        waveform = torch.from_numpy(audio_data.T).float()
        # make sure shape is (1, num_samples) for mono audio. take the mean across channels if stereo.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # If segment is shorter than target_frames (e.g. last segment), pad with zeros on the right
        if waveform.shape[-1] < target_frames:
            padding = target_frames - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return AudioSample(waveform=waveform, sample_rate=sample_rate)


if __name__ == "__main__":
    import numpy as np
    import tempfile

    # Create 2 dummy wav files: 10 seconds and 25 seconds at 16kHz
    sr = 16000
    wav1 = np.random.randn(sr * 10, 1).astype(np.float32)
    wav2 = np.random.randn(sr * 25, 1).astype(np.float32)

    with (
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1,
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2,
    ):
        sf.write(tmp1.name, wav1, sr)
        sf.write(tmp2.name, wav2, sr)
        path1, path2 = tmp1.name, tmp2.name

    try:
        dummy_dataset = [{"audio_path": path1}, {"audio_path": path2}]
        # 10s audio + 25s audio with subset_size=1.0 (1 second per segment)
        segmented_ds = SegmentedAudioDataset(dummy_dataset, subset_size=1.0)
        print(f"Original dataset len: {len(dummy_dataset)}")
        print(f"Segmented dataset len: {len(segmented_ds)}")  # Should be 10 + 25 = 35
        assert len(segmented_ds) == 35

        sample = segmented_ds[0]
        print("Segment 0 waveform shape:", sample["waveform"].shape)
        assert sample["waveform"].shape == (1, 16000)
        print("SegmentedAudioDataset test passed successfully!")
    finally:
        for p in (path1, path2):
            if os.path.exists(p):
                os.remove(p)
