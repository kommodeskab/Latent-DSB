from src import AudioSample
from src.datasets import BaseDataset


class AudioDataset(BaseDataset):
    def __getitem__(self, index: int) -> AudioSample: ...
