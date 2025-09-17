"""
Heuristic dereverberation using WPE.
Based on https://github.com/fgnt/nara_wpe/tree/master
and https://www.audiolabs-erlangen.de/media/pages/resources/aps-w23/papers/259461a00d-1663358899/sap_Yoshioka2012.pdf
"""

from nara_wpe.utils import stft, istft, get_stft_center_frequencies
import torch
from nara_wpe.wpe import wpe

class WPE:
    def __init__(self):
        self.taps = 10
        self.delay = 3
        self.iterations = 5
        self.stft_size=512
        self.stft_shift=128
        
    def encode(self, x : torch.Tensor) -> torch.Tensor: return x
    def decode(self, x : torch.Tensor) -> torch.Tensor: return x
    def to(self, device : str) -> None: ...
    
    def dereverb(self, audio : torch.Tensor) -> torch.Tensor:
        device = audio.device
        processed = []
        
        for frame in audio:
            frame = frame.cpu().numpy()
            Y = stft(frame, self.stft_size, self.stft_shift)
            Y = Y.transpose(2, 0, 1)
            Z = wpe(Y, taps=self.taps, delay=self.delay, iterations=self.iterations, statistics_mode='full')
            dereverb = istft(Z.transpose(1, 2, 0), size=self.stft_size, shift=self.stft_shift)
            processed.append(torch.tensor(dereverb, device=device))

        processed = torch.stack(processed).to(device=device, dtype=torch.float32)
        return processed