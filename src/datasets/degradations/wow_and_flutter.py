import torch
from torch import Tensor
import torch.nn.functional as F
from src.datasets.degradations import BaseDegradation

class WowAndFlutter(BaseDegradation):
    """
    Emulates pitch instability caused by speed fluctuations in the turntable motor or a 
    warped or off center record.
    - Wow: Low-frequency (0.5 - 2 Hz) cyclic pitch warping. Caused by warping or heavy motor lugging
    - Flutter: Higher-frequency (6 - 20 Hz) erratic pitch jitter. Caused by friction or slippage
    """

    def __init__(
        self,
        wow_freq: float = 0.55,    # 33.3 RPM / 60 = 0.55 Hz
        wow_depth: float = 0.002,   # 0.2% pitch variation
        flutter_freq: float = 5.0,
        flutter_depth: float = 0.001, # 0.1% pitch variation
        sample_rate: int = 16000,
        prob: float = 1.0,
        deterministic: bool = False,
    ):
        super().__init__(prob=prob, deterministic=deterministic)
        self.wow_freq = wow_freq
        self.wow_depth = wow_depth
        self.flutter_freq = flutter_freq
        self.flutter_depth = flutter_depth
        self.sample_rate = sample_rate

    def fun(self, audio: Tensor) -> Tensor:
            
        device = audio.device
        num_samples = audio.shape[-1]
        t = torch.arange(num_samples, device=device) / self.sample_rate

        # 1. Generate Modulation (LFOs)
        # Wow is usually a Sine wave (periodic warp)
        wow_lfo = torch.sin(2 * torch.pi * self.wow_freq * t) * self.wow_depth
        
        # Flutter is usually more erratic (Noise-modulated)
        # We use a smoothed random signal for flutter
        flutter_lfo = torch.sin(2 * torch.pi * self.flutter_freq * t) * self.flutter_depth
        
        # Combined displacement in samples
        # Displacement(t) = t + wow(t) + flutter(t)
        total_mod = wow_lfo + flutter_lfo
        
        # 1. Create the 1D indices (x-coordinates)
        indices = torch.arange(num_samples, device=device) + (total_mod * self.sample_rate)
        grid_x = (indices / (num_samples - 1) * 2) - 1

        # 2. Create a dummy y-coordinate (zeros)
        grid_y = torch.zeros_like(grid_x)

        # 3. Stack them to get [num_samples, 2]
        grid = torch.stack((grid_x, grid_y), dim=-1)

        # 4. Reshape to [Batch, Height, Width, 2] -> [1, 1, num_samples, 2]
        grid = grid.view(1, 1, num_samples, 2)

        # 5. Prepare audio: [Batch, Channels, Height, Width] -> [1, Channels, 1, num_samples]
        input_audio = audio.view(1, -1, 1, num_samples)

        output = F.grid_sample(input_audio, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return output.squeeze().unsqueeze(0) # Back to original shape