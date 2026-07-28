from dotenv import load_dotenv
import math
import os
from pathlib import Path
import sys
import gdown
import scipy.signal
import torch
from src.lightning_modules.baselines.sgmse.sgmse.model import ScoreModel
from src.lightning_modules.baselines.sgmse.sgmse.util.other import pad_spec
from src.lightning_modules import BaseLightningModule
from torch import Tensor

# Add sgmse to path before loading checkpoint
sgmse_path = Path(__file__).parent / "sgmse"
sys.path.insert(0, str(sgmse_path.parent))


load_dotenv()

# Make sgmse discoverable for checkpoint unpickling
_baselines_path = os.path.join(os.path.dirname(__file__), "..")
if _baselines_path not in sys.path:
    sys.path.insert(0, _baselines_path)


def resample_audio(x: Tensor, orig_sr: int, target_sr: int) -> Tensor:
    if orig_sr == target_sr:
        return x
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    device = x.device
    dtype = x.dtype
    x_np = x.detach().cpu().numpy()
    resampled_np = scipy.signal.resample_poly(x_np, up, down, axis=-1)
    return torch.from_numpy(resampled_np).to(device=device, dtype=dtype)


class SGMSE(BaseLightningModule):
    def __init__(self, task: str = "rir"):
        super().__init__()
        self.task = task.lower()

        data_path = os.getenv("DATA_PATH", "/home/andbagge/data")
        if self.task == "rir":
            output = f"{data_path}/sgmse_natural_rir_350k.ckpt"
            url = "https://drive.google.com/uc?id=1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD"
        elif self.task == "noise":
            output = f"{data_path}/sgmse_ears_wham.ckpt"
            url = "https://drive.google.com/uc?id=1t_DLLk8iPH6nj8M5wGeOP3jFPaz3i7K5"
        else:
            raise ValueError(f"Unknown task: {self.task}. Must be 'rir' or 'noise'.")

        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        self.model = ScoreModel.load_from_checkpoint(output, strict=True, weights_only=False).cpu()
        self.model.eval()

    def common_step(self, batch, batch_idx):
        return ...

    def sample(self, x_start: Tensor, num_steps: int, **kwargs) -> Tensor:
        T_orig = x_start.size(-1)

        if self.task == "noise" or getattr(self.model, "backbone", "") == "ncsnpp_48k":
            x_proc = resample_audio(x_start, 16000, 48000)
        else:
            x_proc = x_start

        T_proc = x_proc.size(-1)
        norm_factor = x_proc.abs().max()
        if norm_factor > 0:
            x_proc = x_proc / norm_factor
        else:
            norm_factor = 1.0

        Y = [self.model._forward_transform(self.model._stft(x)) for x in x_proc]
        Y = torch.stack(Y, dim=0)

        pad_mode = "reflection" if getattr(self.model, "backbone", "") in ("ncsnpp_48k", "ncsnpp_v2") else "zero_pad"
        Y = pad_spec(Y, mode=pad_mode)

        sampler = self.model.get_pc_sampler(
            "reverse_diffusion",
            "ald",
            Y,
            N=num_steps,
            corrector_steps=1,
            snr=0.5,
        )

        sample, _ = sampler()
        sample = torch.stack([self.model.to_audio(s, T_proc) for s in sample], 0)
        sample = sample * norm_factor

        if self.task == "noise" or getattr(self.model, "backbone", "") == "ncsnpp_48k":
            sample = resample_audio(sample, 48000, 16000)
            if sample.size(-1) > T_orig:
                sample = sample[..., :T_orig]
            elif sample.size(-1) < T_orig:
                sample = torch.nn.functional.pad(sample, (0, T_orig - sample.size(-1)))

        return sample


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_rir = SGMSE(task="rir").to(device)
    x = torch.randn(2, 1, 32000).to(device)
    sample_rir = model_rir.sample(x, num_steps=1)
    print("RIR sample shape:", sample_rir.shape)

    model_noise = SGMSE(task="noise").to(device)
    sample_noise = model_noise.sample(x, num_steps=1)
    print("Noise sample shape:", sample_noise.shape)
