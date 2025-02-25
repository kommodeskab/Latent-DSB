from torch.utils.data import Dataset
from .utils import get_data_path
import os
import json
from pathlib import Path
import torchaudio
from typing import Literal
import random
from torch import Tensor
from torchaudio.transforms import Resample
from torch.nn.functional import pad

class EarsGender(Dataset):
    def __init__(self, gender : Literal['male', 'female'], length_seconds : int = 5.1, sample_rate : int = 24_000):
        super().__init__()
        assert gender in ['male', 'female']
        self.length_seconds = length_seconds
        self.sample_rate = sample_rate
        
        data_path = get_data_path()
        data_path = os.path.join(data_path, 'ears')
        stats : dict = json.load(open(os.path.join(data_path, 'speaker_statistics.json')))        
        folder_names = [os.path.join(data_path, k) for k, v in stats.items() if v['gender'] == gender]
        
        self.file_names = []
        banned = ['nonverbal', 'vegetative']
        for folder in folder_names:
            folder = Path(folder)
            for wav in folder.glob('*.wav'):
                if any([b in str(wav) for b in banned]):
                    continue
                self.file_names.append(str(wav))
                        
    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx) -> Tensor:
        file_name = self.file_names[idx]
        info = torchaudio.info(file_name)
        sample_rate, number_frames = info.sample_rate, info.num_frames
        
        wanted_frames = int(self.length_seconds * sample_rate)
        max_offset = max(0, number_frames - wanted_frames)
        offset = random.uniform(0, max_offset)
        waveform, _ = torchaudio.load(file_name, frame_offset=offset, num_frames=wanted_frames)
        
        if waveform.size(1) < wanted_frames:
            waveform = pad(waveform, (0, wanted_frames - waveform.size(1)))
            
        resampler = Resample(sample_rate, self.sample_rate)
        waveform = resampler(waveform)
        
        return waveform