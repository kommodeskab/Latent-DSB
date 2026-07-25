from src import AudioSample, AudioPathSample
from .basedataset import BaseDataset


class AudioDataset(BaseDataset):
    def __getitem__(self, index: int) -> AudioSample: ...


class AudioPathDataset(BaseDataset):
    def __getitem__(self, index: int) -> AudioPathSample: ...
