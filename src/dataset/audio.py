from torch.utils.data import Dataset, ConcatDataset
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
import glob
import torch
import torch.nn.functional as F
from torchaudio.functional import fftconvolve


def compute_average_sdr(clean: torch.Tensor, processed: torch.Tensor) -> float:
    """
    Compute Signal-to-Distortion Ratio
    
    Args:
        clean: Clean reference signal [B, 1, T] or [B, T]
        processed: Processed/estimated signal [B, 1, T] or [B, T]
        
    Returns:
        SDR value in dB (higher is better)
    """
    # Ensure proper dimensions
    if clean.dim() == 3:
        clean = clean.squeeze(1)
    if processed.dim() == 3:
        processed = processed.squeeze(1)
        
    # Calculate scaling factor for optimal alignment
    # (projection of processed signal onto clean signal)
    scalar = torch.sum(clean * processed, dim=1) / (torch.sum(clean ** 2, dim=1) + 1e-8)
    
    # Calculate scaled processed signal
    s_target = scalar.unsqueeze(1) * clean
    
    # Calculate distortion
    e_distortion = s_target - processed
    
    # Calculate SDR
    sdr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=1) + 1e-8) / 
        (torch.sum(e_distortion ** 2, dim=1) + 1e-8)
    )
    
    return sdr.mean().item()

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
    def __init__(
        self, 
        gender : Literal['male', 'female'],
        split : Literal['train', 'test'] = 'train',
        ):
        # we use speakers p001 to p104 for training
        # and p105 to p107 for testing
        super().__init__()
        assert gender in ['male', 'female']
        assert split in ['train', 'test']
        
        test_speakers = ['p105', 'p106', 'p107']
        
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
                
                is_for_test = wav.parent.name in test_speakers
                if (split == 'train' and is_for_test) or (split == 'test' and not is_for_test):
                    continue
                
                self.file_names.append(str(wav))
    
class VoxCeleb(BaseAudioDataset):
    def __init__(self, gender : Literal['male', 'female'], split : Literal['train', 'test']):
        super().__init__()
        assert gender in ['male', 'female']
        assert split in ['train', 'test']
        
        data_path = get_data_path()
        gender = gender + 's'
        folder_name = f'VoxCeleb_gender/{gender}'
        data_path = os.path.join(data_path, folder_name)
        file_names = glob.glob(os.path.join(data_path, '*.m4a'))
        # file_names have form /datapath/VoxCeleb_gender/males/<number>.m4a
        # if <number> is below 200, it is a test file, else it is a train file
        ids = [int(os.path.basename(f).split('.')[0]) for f in file_names]
        if split == 'train':
            self.file_names = [f for f, id in zip(file_names, ids) if id >= 200]
        else:
            self.file_names = [f for f, id in zip(file_names, ids) if id < 200]
            
        
class FSDNoisy18k(BaseAudioDataset):
    def __init__(self, split : Literal['train', 'test']):
        super().__init__()
        data_path = get_data_path()
        data_path = os.path.join(data_path, f'FSDNoisy18k/FSDnoisy18k.audio_{split}')
        self.file_names = glob.glob(os.path.join(data_path, '*.wav'))
        
class VCTK(BaseAudioDataset):
    def __init__(self, split : Literal['train', 'test'], gender : Literal['male', 'female'] = None):
        super().__init__()
        assert split in ['train', 'test']
        assert gender in [None, 'male', 'female']
        
        if gender is not None:
            gender = 'M' if gender == 'male' else 'F'
        data_path = get_data_path()
        wavs_path = os.path.join(data_path, f'VCTK/wav48_silence_trimmed/')
        # randomly chosen speakers for test set
        test_speakers = ['p225', 'p226', 'p227', 'p228', 'p229', 'p230']
 
        speaker_info = os.path.join(data_path, 'VCTK/speaker-info.txt')
        with open(speaker_info, 'r') as f:
            lines = f.readlines()
        
        lines = lines[1:]
        lines = [line.strip() for line in lines]
        lines = [line.split(' ') for line in lines]
        lines = [[char for char in line if char] for line in lines]
        self.gender_dict = {line[0] : line[2] for line in lines}
        
        all_file_names = glob.glob(os.path.join(wavs_path, '**', '*.flac'), recursive=True)
        self.file_names = []
        for file_name in all_file_names:
            speaker_id = file_name.split("/")[-2]
            is_for_test = speaker_id in test_speakers
            
            if (split == 'train' and is_for_test) or (split == 'test' and not is_for_test):
                continue
            
            if gender is not None:
                speaker_gender = self.gender_dict[speaker_id]
                if speaker_gender != gender:
                    continue
                
            self.file_names.append(file_name)    
            
                
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
        subset : Literal['train-clean-100', 'test-clean'],
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
        initial_sample_rate : int | None = None,
        ):
        super().__init__(datasets)
        self.length_seconds = length_seconds
        self.sample_rate = sample_rate
        self.initial_sample_rate = initial_sample_rate or sample_rate
    
    def __getitem__(self, idx) -> Tensor:
        file_name = super().__getitem__(idx)
        try:
            info = torchaudio.info(file_name)
            original_sample_rate, number_frames = info.sample_rate, info.num_frames
            wanted_frames = int(self.length_seconds * original_sample_rate)
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
                
            # first, resample to initial sample rate to ensure all waveforms have same quality
            # then, resample to specific sample rate
            # only resample, if the original sample rate is different
            if original_sample_rate != self.initial_sample_rate:
                waveform = Resample(original_sample_rate, self.initial_sample_rate)(waveform)
            if self.initial_sample_rate != self.sample_rate:
                waveform = Resample(self.initial_sample_rate, self.sample_rate)(waveform)
            
            #normalize
            rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
            scale = 0.1 / rms
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
        sample_rate : int = 16_000,
        train : bool = True,
        initial_sample_rate : int | None = None,
        ):
        if train:  
            datasets = [
                EarsGender(gender, 'train'),
                VoxCeleb(gender, 'train'),
                LibriSpeech(gender, 'train-clean-100'),
                VCTK('train', gender),
            ]
        else:
            datasets = [
                EarsGender(gender, 'test'),
                VoxCeleb(gender, 'test'),
                LibriSpeech(gender, 'test-clean'),
                VCTK('test', gender),
            ]
                
        super().__init__(
            datasets,
            length_seconds,
            sample_rate,
            initial_sample_rate,
        )
        
class SpeechNoiseDataset(Dataset):
    def __init__(
        self, 
        speech_dataset : BaseConcatAudio,
        noise_dataset : BaseConcatAudio,
        return_pair : bool = False,
        **kwargs,
        ):
        self.speech_dataset = speech_dataset
        self.noise_dataset = noise_dataset
        self.return_pair = return_pair
        
        assert speech_dataset.length_seconds == noise_dataset.length_seconds
        assert speech_dataset.sample_rate == noise_dataset.sample_rate
        
        self.min_snr, self.max_snr = -2, 18
        
        super().__init__()
        
    def __len__(self) -> int:
        return len(self.speech_dataset)
    
    def set_length(self, length_seconds : float) -> None:
        self.speech_dataset.length_seconds = length_seconds
        self.noise_dataset.length_seconds = length_seconds
        
    @property
    def length_seconds(self) -> float:
        return self.speech_dataset.length_seconds
    
    @property
    def sample_rate(self) -> int:
        return self.speech_dataset.sample_rate
    
    @property
    def noise_range(self) -> tuple[float, float]:
        return self.min_snr, self.max_snr
    
    @noise_range.setter
    def noise_range(self, value : tuple[float, float]) -> None:
        self.min_snr, self.max_snr = value
        assert self.min_snr <= self.max_snr, "min_snr must be less than or equal max_snr"
    
    @staticmethod
    def calculate_noise_factor(x : Tensor, noise : Tensor, snr : int) -> int:
        x_l2 = x.pow(2).sum()
        noise_l2 = noise.pow(2).sum()
        a = torch.sqrt(x_l2 / noise_l2 * 10 ** (- snr / 10))
        # if a is infinitely large, set it to 0
        if torch.isinf(a) or torch.isnan(a):
            a = 0
        
        return a
    
    def get_item(self, idx, snr : int | None = None) -> tuple[Tensor, Tensor] | Tensor:
        speech = self.speech_dataset[idx]
        if torch.rand(1) < 0.05 and snr is None:
            noisy_speech = speech.clone()
        else:
            noise_idx = torch.randint(0, len(self.noise_dataset), (1,))
            noise = self.noise_dataset[noise_idx]
            random_snr = torch.randint(self.min_snr, self.max_snr, (1,)) if snr is None else snr
            noise_factor = self.calculate_noise_factor(speech, noise, random_snr)
            noisy_speech = speech + noise_factor * noise
        
        if self.return_pair:
            return speech, noisy_speech
        
        return noisy_speech

    def __getitem__(self, idx) -> tuple[Tensor, Tensor] | Tensor:
        return self.get_item(idx)
    
class LibriFSDPaired(SpeechNoiseDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        initial_sample_rate : int | None = None,
        train : bool = True,
        return_pair : bool = False,
    ):
        """
        Returns only the noisy sample. Can be used for unpaired training.        
        """
        self.return_pair = return_pair
        
        if train:
            speech = [LibriSpeech('male', 'train-clean-100'), LibriSpeech('female', 'train-clean-100')]
            noise = [FSDNoisy18k('train')]
        else:
            speech = [LibriSpeech('male', 'test-clean'), LibriSpeech('female', 'test-clean')]
            noise = [FSDNoisy18k('test')]
            
        speech_dataset = BaseConcatAudio(speech, length_seconds, sample_rate, initial_sample_rate)
        noise_dataset = BaseConcatAudio(noise, length_seconds, sample_rate, initial_sample_rate)
        super().__init__(speech_dataset, noise_dataset, return_pair)
        
class VCTKWHAM(SpeechNoiseDataset):
    def __init__(
        self,
        length_seconds : float,
        sample_rate : int = 16_000,
        train : bool = True,
        return_pair : bool = False,
        **kwargs,
    ):
        self.return_pair = return_pair
        
        if train:
            speech = [VCTK('train', gender=None)]
            noise = [WHAM('train')]
        else:
            speech = [VCTK('test', gender=None)]
            noise = [WHAM('test')]
        
        speech_dataset = BaseConcatAudio(speech, length_seconds, sample_rate)
        noise_dataset = BaseConcatAudio(noise, length_seconds, sample_rate)
        super().__init__(speech_dataset, noise_dataset, return_pair)
        
class EarsWHAMUnpaired(SpeechNoiseDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        initial_sample_rate : int | None = None,
        train : bool = True,
        return_pair : bool = False,
        **kwargs,
    ):
        """
        This is a noisy speech datset consisting of Ears and WHAM, but only the noisy speech is returned.
        This is used for unpaired training.
        """
        
        self.return_pair = return_pair
        
        if train:
            speech = [EarsGender('male', 'train'), EarsGender('female', 'train')]
            noise = [WHAM('train')]
        else:
            speech = [EarsGender('male', 'test'), EarsGender('male', 'test')]
            noise = [WHAM('test')]
    
        speech_dataset = BaseConcatAudio(speech, length_seconds, sample_rate, initial_sample_rate)
        noise_dataset = BaseConcatAudio(noise, length_seconds, sample_rate, initial_sample_rate)
        super().__init__(speech_dataset, noise_dataset, return_pair)
        
class AllLibri(BaseConcatAudio):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        initial_sample_rate : int | None = None,
        train : bool = True,
        **kwargs,
        ):
        if train:
            datasets = [
                LibriSpeech('male', 'train-clean-100'),
                LibriSpeech('female', 'train-clean-100'),
            ]
        else:
            datasets = [
                LibriSpeech('male', 'test-clean'),
                LibriSpeech('female', 'test-clean'),
            ]
        
        super().__init__(
            datasets,
            length_seconds,
            sample_rate,
            initial_sample_rate,
        )

class AllVCTK(BaseConcatAudio):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        initial_sample_rate : int | None = None,
        train : bool = True,
        gender = None,
        **kwargs,
        ):
        if train:
            datasets = [VCTK('train', gender),]
        else:
            datasets = [VCTK('test', gender)]
        
        super().__init__(
            datasets,
            length_seconds,
            sample_rate,
            initial_sample_rate,
        )
        
class ClippedDataset(Dataset):
    def __init__(
        self,
        dataset : BaseConcatAudio,
        gain_range : tuple[float, float] = (5, 30),
        return_pair : bool = False,
        no_clip_prob : float = 0.0,
        **kwargs,
    ):
        self.no_clip_prob = no_clip_prob
        self.gain_min, self.gain_max = gain_range
        self.dataset = dataset
        self.return_pair = return_pair
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def set_length(self, length_seconds : float) -> None:
        self.dataset.length_seconds = length_seconds
        
    @property
    def length_seconds(self) -> float:
        return self.dataset.length_seconds
    
    @property
    def sample_rate(self) -> int:
        return self.dataset.sample_rate
    
    def what_db_for_snr(self, target_snr : float) -> float:
        """
        given some desired SNR level, find the gain db in the noise range which gives that SNR.
        Uses binary search.
        """
        old_return_pair = self.return_pair
        self.return_pair = True
        low, high = -10, 50 # start with arbitrary low/high values for the binary search
        while abs(high - low) > 0.01:
            mid = (low + high) / 2
            clean, noisy = self.get_item(0, mid)
            clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
            sdr = compute_average_sdr(clean, noisy)
            if sdr < target_snr:
                high = mid
            else:
                low = mid
                
        self.return_pair = old_return_pair
        return mid
    
    @property
    def noise_range(self) -> tuple[float, float]:
        return self.gain_min, self.gain_max
    
    @noise_range.setter
    def noise_range(self, value : tuple[float, float]) -> None:
        self.gain_min, self.gain_max = value
        assert self.gain_min <= self.gain_max, "gain_min must be less than or equal gain_max"
        
    def get_item(self, idx : int, gain_db : float | None = None) -> tuple[Tensor, Tensor]:
        original : Tensor = self.dataset[idx]
        if random.random() < self.no_clip_prob and gain_db is None:
            clipped = original.clone()
        else:
            clipped = original.clone()
            gain_db = random.uniform(self.gain_min, self.gain_max) if gain_db is None else gain_db
            gain_lim = 10 ** (gain_db / 20)
            clipped = clipped * gain_lim
            clipped = clipped.clamp(-1, 1)
            clipped = clipped / gain_lim
            
        if self.return_pair:
            return original, clipped
        
        return clipped
    
    def __getitem__(self, index : int) -> Tensor | tuple[Tensor, Tensor]:
        return self.get_item(index)
    
class ClippedLibri(ClippedDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        train : bool = True,
        gain_range : tuple[float, float] = (5, 30),
        return_pair : bool = False,
        no_clip_prob : float = 0.0,
    ):
        dataset = AllLibri(length_seconds, sample_rate, train=train)
        super().__init__(
            dataset,
            gain_range=gain_range,
            return_pair=return_pair,
            no_clip_prob=no_clip_prob,
        )

class ClippedVCTK(ClippedDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        train : bool = True,
        gain_range : tuple[float, float] = (5, 30),
        return_pair : bool = False,
        no_clip_prob : float = 0.0,
    ):
        dataset = AllVCTK(length_seconds, sample_rate, train=train)
        super().__init__(
            dataset,
            gain_range=gain_range,
            return_pair=return_pair,
            no_clip_prob=no_clip_prob,
        )
class RIRS(BaseAudioDataset):
    def __init__(self, train : bool = True):
        super().__init__()
        
        data_path = get_data_path()
        if train:
            rirs_paths = [
                "BUT_ReverbDB",
                "OPENAIR",
                "RWCP_REVERB_AACHEN",
                "C4DM",
            ]
        else:
            rirs_paths = [
                "MIT_Survey",
            ]
        
        rirs_paths = [os.path.join(data_path, p) for p in rirs_paths]
        
        self.file_names = []
        for path in rirs_paths:
            wav_files = glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)
            self.file_names.extend(wav_files)

class RIRDataset(Dataset):
    def __init__(
        self,
        audio_dataset : BaseConcatAudio,
        train : bool = True,
        return_pair : bool = False,
    ):
        self.audio_dataset = audio_dataset
        self.rir_dataset = BaseConcatAudio(
            [RIRS(train)],
            length_seconds=5.0,
            sample_rate=audio_dataset.sample_rate,
        )
        self.return_pair = return_pair
        self.sample_rate = audio_dataset.sample_rate
        super().__init__()
        
    def __len__(self) -> int:
        return len(self.audio_dataset)
    
    @staticmethod
    def estimate_rir_delay(rir: torch.Tensor, threshold_db: float = -20.0) -> int:
        rir = rir.squeeze()
        energy = rir.abs()
        max_energy = energy.max()

        # Convert dB threshold to linear scale
        threshold = max_energy * (10 ** (threshold_db / 20.0))

        # Find the first sample exceeding the threshold
        delay_idx = (energy >= threshold).nonzero(as_tuple=False)
        
        if delay_idx.numel() == 0:
            return 0  # fallback if RIR is silent
        return delay_idx[0].item()
    
    def apply_rir(self, audio : Tensor, rir : Tensor) -> Tensor:
        delay = self.estimate_rir_delay(rir)
        
        start = max(0, delay + int(0.01 * self.sample_rate))
        duration = int(0.3 * self.sample_rate)
        end = start + duration
        
        rir = rir[:, start : end]
        rir = rir / torch.norm(rir, p=2)
        
        augmented = fftconvolve(audio, rir, mode='same')
        # normalize the augmented audio since it can be louder than the original audio
        rms = torch.sqrt(torch.mean(augmented**2) + 1e-8)
        scale = 0.1 / rms
        augmented = scale * augmented
        
        return augmented
    
    def get_item(self, index : int):
        # used for compatibility with ClippedDataset
        return self.__getitem__(index)
    
    def __getitem__(self, index: int):
        audio = self.audio_dataset[index]
        rand_idx = torch.randint(0, len(self.rir_dataset), (1,)).item()
        rir = self.rir_dataset[rand_idx]
        
        augmented = self.apply_rir(audio, rir)
        if self.return_pair:
            return audio, augmented
        
        return augmented
    
class VCTKRIR(RIRDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        train : bool = True,
        return_pair : bool = False,
    ):
        audio_dataset = AllVCTK(
            length_seconds=length_seconds,
            sample_rate=sample_rate,
            train=train,
        )
        
        super().__init__(audio_dataset, train, return_pair)

class LibriRIR(RIRDataset):
    def __init__(
        self,
        length_seconds : float = 5.1,
        sample_rate : int = 16_000,
        train : bool = True,
        return_pair : bool = False,
    ):
        audio_dataset = AllLibri(
            length_seconds=length_seconds,
            sample_rate=sample_rate,
            train=train,
        )
        
        super().__init__(audio_dataset, train, return_pair)

class LibriWHAM(SpeechNoiseDataset):
    def __init__(
        self,
        length_seconds : float,
        sample_rate : int = 16_000,
        train : bool = True,
        return_pair : bool = False,
        **kwargs,
    ):
        self.return_pair = return_pair
        
        if train:
            speech = [LibriSpeech('male', 'train-clean-100'), LibriSpeech('female', 'train-clean-100')]
            noise = [WHAM('train')]
        else:
            speech = [LibriSpeech('male', 'test-clean'), LibriSpeech('female', 'test-clean')]
            noise = [WHAM('test')]
        
        speech_dataset = BaseConcatAudio(speech, length_seconds, sample_rate)
        noise_dataset = BaseConcatAudio(noise, length_seconds, sample_rate)
        super().__init__(speech_dataset, noise_dataset, return_pair)