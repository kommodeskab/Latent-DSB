from torch.utils.data import Dataset, ConcatDataset
from .utils import get_data_path
import os
import json
from pathlib import Path
import torchaudio
from typing import Literal
import random
from torch import Tensor
from torchaudio.transforms import Resample, Vol, PitchShift
from torch.nn.functional import pad
import glob
import torch
import time

class BaseAudioDataset(Dataset):
    file_names : list[str]
    
    def __init__(self):
        super().__init__()
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx) -> str:
        file_name = self.file_names[idx]
        return file_name

class EarsGender(BaseAudioDataset):
    def __init__(self, gender : Literal['male', 'female']):
        super().__init__()
        assert gender in ['male', 'female']
        
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
    
class VoxCeleb(BaseAudioDataset):
    def __init__(self, gender : Literal['male', 'female']):
        super().__init__()
        data_path = get_data_path()
        gender = gender + 's'
        folder_name = f'VoxCeleb_gender/{gender}'
        data_path = os.path.join(data_path, folder_name)
        self.file_names = glob.glob(os.path.join(data_path, '*.m4a'))
    
class GenderAudioDataset(ConcatDataset):
    def __init__(self, gender : Literal['male', 'female'], length_seconds : float = 5.1, sample_rate : int = 24_000):
        self.length_seconds = length_seconds
        self.sample_rate = sample_rate
        
        earsdataset = EarsGender(gender)
        vox = VoxCeleb(gender)
        super().__init__([earsdataset, vox])
        
    def __getitem__(self, idx) -> Tensor:
        file_name = super().__getitem__(idx)
        info = torchaudio.info(file_name)
        sample_rate, number_frames = info.sample_rate, info.num_frames
        
        wanted_frames = int(self.length_seconds * sample_rate)
        max_offset = max(0, number_frames - wanted_frames)
        offset = random.uniform(0, max_offset)
        waveform, _ = torchaudio.load(file_name, frame_offset=offset, num_frames=wanted_frames)
        
        if waveform.size(1) < wanted_frames:
            waveform = pad(waveform, (0, wanted_frames - waveform.size(1)))
            
        # resmaple to specified sample rate
        waveform = Resample(sample_rate, self.sample_rate)(waveform)
        
        # randomly increase or decrease volume
        random_gain = random.uniform(0.5, 1.5)
        waveform = Vol(random_gain)(waveform)
                
        return waveform