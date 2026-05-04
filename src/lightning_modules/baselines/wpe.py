from nara_wpe.utils import stft, istft
import torch
from nara_wpe.wpe import wpe
from src.lightning_modules import BaseLightningModule


class WPE(BaseLightningModule):
    """
    Weighted Prediction Error for speech dereverberation
    See https://github.com/fgnt/nara_wpe
    Used as a baseline
    """

    def __init__(self):
        super().__init__()
        self.taps = 20
        self.delay = 3
        self.iterations = 5
        self.stft_size = 512
        self.stft_shift = 128

    def common_step(self, batch, batch_idx):
        return ...

    def forward(self, x_start: torch.Tensor, **kwargs) -> torch.Tensor:
        device = x_start.device
        processed = []

        for frame in x_start:
            frame = frame.cpu().numpy()
            Y = stft(frame, self.stft_size, self.stft_shift)
            Y = Y.transpose(2, 0, 1)
            Z = wpe(Y, taps=self.taps, delay=self.delay, iterations=self.iterations, statistics_mode="full")
            dereverb = istft(Z.transpose(1, 2, 0), size=self.stft_size, shift=self.stft_shift)
            processed.append(torch.tensor(dereverb, device=device))

        processed = torch.stack(processed).to(device=device, dtype=torch.float32)
        return processed

    def sample(self, x_start: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(x_start,**kwargs)
