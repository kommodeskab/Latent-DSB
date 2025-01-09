from torch.utils.data import Dataset, ConcatDataset
import torchaudio
import os
import torch
import random
from typing import Tuple

class AudioCrawler(Dataset):
    def __init__(self, path : str):
        super().__init__()
        self.path = path
        self.files = os.listdir(path)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        filename = self.files[idx]
        waveform, sample_rate = torchaudio.load(os.path.join(self.path, filename))
        return waveform, sample_rate
    
class ConcatFormattedAudio(ConcatDataset):
    def __init__(
        self, 
        datasets : list[Dataset],
        audio_length : int = 2,
        sample_rate : int = 16000,
        ):
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        super().__init__(datasets)
        
    def __getitem__(self, idx):
        waveform, old_sample_rate = super().__getitem__(idx)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = torchaudio.transforms.Resample(old_sample_rate, self.sample_rate)(waveform)
        
        seq_length = waveform.shape[1]
        new_seq_length = self.audio_length * self.sample_rate
        if seq_length > new_seq_length:
            waveform = waveform[:, :new_seq_length]
        else:
            padding_size = new_seq_length - seq_length
            waveform = torch.nn.functional.pad(waveform, (0, padding_size))
        
        return waveform
    
class Noise(ConcatFormattedAudio):
    def __init__(self, **kwargs):
        paths = [
            "data/MS-SNSD/noise_train",
            "data/MS-SNSD/noise_test",
            "data/FSDNoisy18k/FSDnoisy18k.audio_test",
            "data/FSDNoisy18k/FSDnoisy18k.audio_train",
        ]
        datasets = [AudioCrawler(path) for path in paths]
        super().__init__(datasets, **kwargs)
        
class Clean(ConcatFormattedAudio):
    def __init__(self, **kwargs):
        paths = [
            "data/MS-SNSD/clean_train",
            "data/MS-SNSD/clean_test",
        ]
        datasets = [AudioCrawler(path) for path in paths]
        super().__init__(datasets, **kwargs)
        
class NoisySpeech(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.noise = Noise(**kwargs)
        self.clean = Clean(**kwargs)
        
    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, idx):
        clean = self.clean[idx]
        noise_idx = random.randint(0, len(self.noise) - 1)
        noise = self.noise[noise_idx]
        # if the noise is all zeros, just generate some random noise
        if noise.abs().sum() < 1e-3:
            noise = torch.randn_like(clean)
        
        alpha = random.uniform(0.5, 1.5)
        return clean + alpha * noise
    
if __name__ == "__main__":
    dataset = NoisySpeech()
    for i in range(5):
        waveform = dataset[i]
        print(waveform.shape)
        torchaudio.save(f"test_{i}.wav", waveform, 16000)