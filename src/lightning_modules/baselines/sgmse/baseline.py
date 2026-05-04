from dotenv import load_dotenv
import os
from pathlib import Path
import sys
import gdown
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


class SGMSE(BaseLightningModule):
    def __init__(self):
        super().__init__()

        data_path = os.getenv("DATA_PATH")
        output = f"{data_path}/sgmse_natural_rir_350k.ckpt"
        url = "https://drive.google.com/uc?id=1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD"

        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        self.model = ScoreModel.load_from_checkpoint(output, strict=True, weights_only=False).cpu()
        self.model.eval()

    def common_step(self, batch, batch_idx):
        return ...

    def sample(self, x_start: Tensor, num_steps: int, **kwargs) -> Tensor:
        T_orig = x_start.size(-1)
        norm_factor = x_start.abs().max()
        x_start = x_start / norm_factor
        Y = [self.model._forward_transform(self.model._stft(x)) for x in x_start]
        Y = torch.stack(Y, dim=0)
        Y = pad_spec(Y, mode="zero_pad")

        sampler = self.model.get_pc_sampler(
            "reverse_diffusion",
            "ald",
            Y,
            N=num_steps,
            correcter_steps=1,
            snr=0.5,
        )

        sample, _ = sampler()
        sample = torch.stack([self.model.to_audio(s, T_orig) for s in sample], 0)
        sample = sample * norm_factor

        return sample


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGMSE().to(device)
    x = torch.randn(2, 1, 32000).to(device)
    sample = model.sample(x, num_steps=1)
    print(sample.shape)
