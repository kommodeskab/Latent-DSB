from pathlib import Path
import sys

# Add open_universe directory to sys.path before importing open_universe components
universe_dir = Path(__file__).parent / "open_universe"
if str(universe_dir) not in sys.path:
    sys.path.insert(0, str(universe_dir))

from dotenv import load_dotenv  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torchaudio.functional import resample  # noqa: E402

from open_universe.inference_utils.model_loader import load_model  # noqa: E402
from src.lightning_modules import BaseLightningModule  # noqa: E402

load_dotenv()


class UniversePlusPlus(BaseLightningModule):
    """
    UNIVERSE++ baseline model from https://github.com/line/open-universe
    """

    def __init__(self, model_id: str = "line-corporation/open-universe:plusplus"):
        super().__init__()
        self.model_id = model_id

        # Load model using open_universe model_loader
        self.model = load_model(self.model_id, device="cpu")
        self.model.eval()

    def common_step(self, batch, batch_idx):
        return ...

    def sample(self, x_start: Tensor, num_steps: int = 1, **kwargs) -> Tensor:
        T_orig = x_start.size(-1)
        device = x_start.device

        target_fs = getattr(self.model, "fs", 16000)
        orig_fs = 16000

        self.model = self.model.to(device)

        if orig_fs != target_fs:
            x_proc = resample(x_start, orig_fs, target_fs)
        else:
            x_proc = x_start

        with torch.no_grad():
            enhanced = self.model.enhance(x_proc)

        if orig_fs != target_fs:
            enhanced = resample(enhanced, target_fs, orig_fs)

        if enhanced.size(-1) > T_orig:
            enhanced = enhanced[..., :T_orig]
        elif enhanced.size(-1) < T_orig:
            enhanced = torch.nn.functional.pad(enhanced, (0, T_orig - enhanced.size(-1)))

        return enhanced


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversePlusPlus().to(device)
    x = torch.randn(2, 1, 32000).to(device)
    sample = model.sample(x)
    print("UNIVERSE++ sample shape:", sample.shape)
