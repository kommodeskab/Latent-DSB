from torch.utils.data import Dataset, ConcatDataset
from .utils import get_data_path
import os
import json
from pathlib import Path
import torchaudio
from typing import Literal
import random
from torch import Tensor
from torchaudio.transforms import Resample, Vol
from torch.nn.functional import pad
import glob
import pandas as pd
import torch

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
        
class FSDNoisy18k(BaseAudioDataset):
    def __init__(self, split : Literal['train', 'test']):
        super().__init__()
        data_path = get_data_path()
        data_path = os.path.join(data_path, f'FSDNoisy18k/FSDnoisy18k.audio_{split}')
        self.file_names = glob.glob(os.path.join(data_path, '*.wav'))
        
class WHAM(BaseAudioDataset):
    def __init__(self, split : Literal['train', 'validation', 'test']):
        super().__init__()
        assert split in ['train', 'validation', 'test']
        if split == 'train':
            split = 'tr'
        elif split == 'validation':
            split = 'cv'
        elif split == 'test':
            split = 'tt'
        
        data_path = get_data_path()
        data_path = os.path.join(data_path, f'wham_noise/{split}')
        self.file_names = glob.glob(os.path.join(data_path, '*.wav'))
        
class LibriSpeech(BaseAudioDataset):
    def __init__(
        self, 
        gender : Literal['male', 'female'], 
        subset : str = 'train-clean-100',
        ):
        super().__init__()
        assert gender in ['male', 'female']
        data_path = get_data_path()
        data_path = os.path.join(data_path, 'LibriSpeech')
        txt_path = os.path.join(data_path, 'SPEAKERS.TXT')
        
        with open(txt_path) as f:
            lines = f.readlines()
            
        lines = [l for l in lines if not l.startswith(';')]
        lines = [l.replace(' ', '') for l in lines]
        lines = [l.split('|') for l in lines]
        lines = [l for l in lines if l[2] == subset]
        
        gender_str = 'M' if gender == 'male' else 'F'
        lines = [l for l in lines if l[1] == gender_str]
        ids = [l[0] for l in lines]
        
        self.file_names = []
        for id in ids:
            folder = os.path.join(data_path, subset, id)
            files = glob.glob(os.path.join(folder, '**', '*.flac'), recursive=True)
            self.file_names.extend(files)  
        
class BaseConcatAudio(ConcatDataset):
    def __init__(
        self, 
        datasets : list[Dataset],
        length_seconds : float,
        sample_rate : int,
        ):
        super().__init__(datasets)
        self.length_seconds = length_seconds
        self.sample_rate = sample_rate
    
    def __getitem__(self, idx) -> Tensor:
        file_name = super().__getitem__(idx)
        try:
            info = torchaudio.info(file_name)
            sample_rate, number_frames = info.sample_rate, info.num_frames
            
            wanted_frames = int(self.length_seconds * sample_rate)
            max_offset = max(0, number_frames - wanted_frames)
            offset = random.uniform(0, max_offset)
            waveform, _ = torchaudio.load(file_name, frame_offset=offset, num_frames=wanted_frames)
            waveform = waveform.mean(0, keepdim=True)
            
            if waveform.size(1) < wanted_frames:
                # randomly pad both sides
                pad_amount = wanted_frames - waveform.size(1)
                left_pad = random.randint(0, pad_amount)
                right_pad = pad_amount - left_pad
                waveform = pad(waveform, (left_pad, right_pad))
                
            # resmaple to specified sample rate
            waveform = Resample(sample_rate, self.sample_rate)(waveform)
            
            #normalize
            rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
            scale = 0.2 / rms
            waveform = scale * waveform
        except:
            print(f'Error loading {file_name}. Defaulting to zeros.')
            waveform = torch.zeros(1, int(self.length_seconds * self.sample_rate))
        
        return waveform
          
class GenderAudioDataset(BaseConcatAudio):
    def __init__(
        self, 
        gender : Literal['male', 'female'], 
        length_seconds : float = 5.1, 
        sample_rate : int = 24_000
        ):
        datasets = [
            EarsGender(gender),
            VoxCeleb(gender),
            LibriSpeech(gender),
        ]
        super().__init__(
            datasets,
            length_seconds,
            sample_rate,
        )
        
class SpeechNoiseDataset(Dataset):
    def __init__(
        self, 
        speech_dataset : BaseConcatAudio,
        noise_dataset : BaseConcatAudio,
        flip : bool = False,
        ):
        if flip:
            speech_dataset, noise_dataset = noise_dataset, speech_dataset
        
        self.speech_dataset = speech_dataset
        self.noise_dataset = noise_dataset
        
        assert speech_dataset.length_seconds == noise_dataset.length_seconds
        assert speech_dataset.sample_rate == noise_dataset.sample_rate
        
        self.length_seconds = speech_dataset.length_seconds
        self.sample_rate = speech_dataset.sample_rate
        
        super().__init__()
        
    def __len__(self) -> int:
        return len(self.speech_dataset)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        speech = self.speech_dataset[idx]
        noise_idx = torch.randint(0, len(self.noise_dataset), (1,)).item()
        noise = self.noise_dataset[noise_idx]
        random_snr = torch.randint(-10, 20, (1,))
        noisy_speech = torchaudio.functional.add_noise(speech, noise, snr=random_snr)
        return speech, noisy_speech

class LibriWhamPaired(SpeechNoiseDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 24_000,
        flip : bool = False,
        ):
        speech = [
            LibriSpeech('male'),
            LibriSpeech('female'),
        ]
        noise = [
            WHAM('train'),
            WHAM('validation'),
            WHAM('test'),
        ]
        speech_dataset = BaseConcatAudio(speech, length_seconds, sample_rate)
        noise_dataset = BaseConcatAudio(noise, length_seconds, sample_rate)
        super().__init__(speech_dataset, noise_dataset, flip)
    
class EarsFSDNoisy(SpeechNoiseDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 24_000,
    ):
        """
        Returns only the noisy sample. Can be used for unpaired training.        
        """
        speech = [
            EarsGender('male'),
            EarsGender('female'),
        ]
        noise = [
            FSDNoisy18k('train'),
            FSDNoisy18k('test'),
        ]
        speech_dataset = BaseConcatAudio(speech, length_seconds, sample_rate)
        noise_dataset = BaseConcatAudio(noise, length_seconds, sample_rate)
        super().__init__(speech_dataset, noise_dataset)
        
    def __getitem__(self, idx : int) -> Tensor:
        _, n = super().__getitem__(idx)
        return n
    
class AllLibri(BaseConcatAudio):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 24_000,
        ):
        datasets = [
            LibriSpeech('male'),
            LibriSpeech('female'),
        ]
        super().__init__(
            datasets,
            length_seconds,
            sample_rate,
        )
        
class AllVoxCeleb(BaseConcatAudio):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 24_000,
        ):
        datasets = [
            VoxCeleb('male'),
            VoxCeleb('female'),
        ]
        super().__init__(
            datasets,
            length_seconds,
            sample_rate,
        )

class AllEars(BaseConcatAudio):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 24_000,
        ):
        datasets = [
            EarsGender('male'),
            EarsGender('female'),
        ]
        super().__init__(
            datasets,
            length_seconds,
            sample_rate,
        )