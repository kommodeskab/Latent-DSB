from .audio import BaseAudioDataset, BaseConcatAudio, SpeechNoiseDataset
from .utils import get_data_path
import glob
from typing import Literal

class LombardGridDataset(BaseAudioDataset):
    def __init__(self, split: Literal['train', 'validation']):
        super().__init__()
        # load all the lombard file names into a list
        # the files are stored at "data_path/lombard_grid/audio/"
        data_path = get_data_path()
        file_names = glob.glob(f"{data_path}/lombard_grid/audio/*.wav")
        n_files = len(file_names)
        train_len = int(0.9 * n_files)
        
        if split == 'train':
            self.file_names = file_names[:train_len]
        else:
            self.file_names = file_names[train_len:]

class LombardNijmegenDataset(BaseAudioDataset):
    def __init__(self, split: Literal['train', 'validation']):
        super().__init__()
        data_path = get_data_path()
        self.file_names = glob.glob(f"{data_path}/*lombard.wav")
        if split == 'train':
            self.file_names = self.file_names[:int(0.9 * len(self.file_names))]
        else:
            self.file_names = self.file_names[int(0.9 * len(self.file_names)):]
        

class AllLombardDataset(BaseConcatAudio):
    def __init__(
        self,
        length_seconds: float = 5.0,
        sample_rate: int = 16000,
        split: Literal['train', 'validation'] = 'train'
    ):
        datasets = [
            LombardGridDataset(split=split),
            LombardNijmegenDataset(split=split)
        ]
        super().__init__(
            datasets=datasets,
            length_seconds=length_seconds,
            sample_rate=sample_rate
        )