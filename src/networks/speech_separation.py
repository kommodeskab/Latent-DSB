import torch
from torch import Tensor
from torchaudio.functional import resample
import torch.nn as nn
from speechbrain.inference.separation import SepformerSeparation as separator

class SpeechbrainSepformer(nn.Module):
    def __init__(self, source : str = "sepformer-wham"):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source = source
        self.model = separator.from_hparams(
            source=f'speechbrain/{source}', 
            savedir=f'pretrained_models/{source}',
            run_opts={"device": device}
        )
        self.model.eval()
        self.sample_rate = self.model.hparams.sample_rate
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded SpeechbrainSepformer from {source} with {n_params} parameters.")
        
        vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (get_speech_timestamps, _, _, _, _) = utils
        self.vad = vad
        self.get_speech_timestamps = get_speech_timestamps
        
    def get_speech(self, separated : Tensor, sample_rate : int = 16000) -> Tensor:
        c1, c2 = separated[:, :, 0], separated[:, :, 1]
        c1_timestamps = torch.tensor([len(self.get_speech_timestamps(s, self.vad, sampling_rate=sample_rate)) for s in c1])
        c2_timestamps = torch.tensor([len(self.get_speech_timestamps(s, self.vad, sampling_rate=sample_rate)) for s in c2])
        # extract speech as the channel with the most speech
        s = c1
        mask = c2_timestamps > c1_timestamps
        s[mask] = c2[mask]
        return s
        
    def encode(self, x : Tensor) -> Tensor: return x
    def decode(self, x : Tensor) -> Tensor: return x
        
    @torch.no_grad()
    def separate(self, mixture : Tensor, sample_rate : int) -> Tensor:        
        mixture = resample(mixture, orig_freq=sample_rate, new_freq=self.sample_rate)
        if mixture.dim() == 3:
            mixture = mixture.mean(dim=1)
            
        out = self.model.separate_batch(mixture)
        s = self.get_speech(out, sample_rate=sample_rate)

        s = resample(s, orig_freq=self.sample_rate, new_freq=sample_rate)
        s = s.unsqueeze(1)

        return s