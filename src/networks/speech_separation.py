import torch
from torch import Tensor
from torchaudio.functional import resample
import torch.nn as nn
from speechbrain.inference.separation import SepformerSeparation as separator

class VADClassifier(nn.Module):
    def __init__(
        self,
        sample_rate : int,
        ):
        super().__init__()
        vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.sample_rate = sample_rate
        (get_speech_timestamps, _, _, _, _) = utils
        self.vad = vad
        self.get_speech_timestamps = get_speech_timestamps
        
    def separate_audio(self, channel1 : Tensor, channel2 : Tensor, sample_rate : int) -> Tensor:
        assert channel1.shape == channel2.shape, "Input tensors must have the same shape"
        assert channel1.dim() == 2, "Expects a batch of 1d tensors as input"
        batch_size = channel1.shape[0]
        output = []
        for i in range(batch_size):
            c1, c2 = channel1[i], channel2[i]
            c1_duration = self.calculate_speech_duration(c1)
            c2_duration = self.calculate_speech_duration(c2)
            speech = c1 if c1_duration >= c2_duration else c2
            output.append(speech)
            
        output = torch.stack(output)
        return output

    def calculate_speech_duration(self, audio : Tensor) -> float:
        assert audio.dim() == 1, "Expects 1d tensor as input"
        timestamps = self.get_speech_timestamps(audio, self.vad, sampling_rate=self.sample_rate)
        
        total_duration = 0.0
        for ts in timestamps:
            total_duration += (ts['end'] - ts['start']) / self.sample_rate

        return total_duration
    
class SpeechbrainSepformer(nn.Module):
    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source = "sepformer-wham16k-enhancement"
        self.model = separator.from_hparams(
            source=f'speechbrain/{self.source}', 
            savedir=f'pretrained_models/{self.source}',
            run_opts={"device": device}
        )
        self.model.eval()
        self.sample_rate = self.model.hparams.sample_rate
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded SpeechbrainSepformer from {self.source} with {n_params} parameters.")
        
    def encode(self, x : Tensor) -> Tensor: return x
    def decode(self, x : Tensor) -> Tensor: return x
        
    @torch.no_grad()
    def separate(self, mixture : Tensor) -> Tensor:
        if self.model.training:
            self.model.eval()
              
        if mixture.dim() == 3:
            mixture = mixture.mean(dim=1)
            
        s = self.model.separate_batch(mixture).squeeze(-1).unsqueeze(1)
        # normalize audio to -1 to 1
        max_val = s.abs().amax(dim=-1, keepdim=True)
        s = s / (max_val + 1e-8)

        return s
    
from asteroid.models import ConvTasNet

class AsteroidConvTasNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = 'cankeles/ConvTasNet_WHAMR_enhsingle_16k'
        self.model = ConvTasNet.from_pretrained(self.source)
        self.model.eval()
        self.sample_rate = int(self.model.sample_rate)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded AsteroidConvTasNet from {self.source} with {n_params} parameters.")
        
    def encode(self, x : Tensor) -> Tensor: return x
    def decode(self, x : Tensor) -> Tensor: return x
        
    @torch.no_grad()
    def separate(self, mixture : Tensor) -> Tensor:   
        if self.model.training:
            self.model.eval()
             
        if mixture.dim() == 3:
            mixture = mixture.mean(dim=1)
            
        s : Tensor = self.model(mixture)
        max_val = s.abs().amax(dim=-1, keepdim=True)
        s = s / (max_val + 1e-8)
        
        return s