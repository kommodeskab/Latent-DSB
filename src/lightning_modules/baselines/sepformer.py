import os
import pytorch_lightning as pl
from torch import Tensor
from dotenv import load_dotenv

import ruamel.yaml
import huggingface_hub
from huggingface_hub.errors import HfHubHTTPError

# Fix compatibility between ruamel.yaml >= 0.19 and hyperpyyaml
ruamel.yaml.Loader.max_depth = None

# Fix compatibility between huggingface_hub >= 0.25 and speechbrain
_orig_hf_hub_download = huggingface_hub.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        val = kwargs.pop("use_auth_token")
        if val and "token" not in kwargs:
            kwargs["token"] = val
    try:
        return _orig_hf_hub_download(*args, **kwargs)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            raise ValueError("File not found on HF hub") from e
        raise


huggingface_hub.hf_hub_download = _patched_hf_hub_download

from speechbrain.inference.separation import SepformerSeparation  # noqa: E402

load_dotenv()

data_path = os.getenv("DATA_PATH")


class Sepformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_name = "speechbrain/sepformer-wham16k-enhancement"
        self.model: SepformerSeparation = SepformerSeparation.from_hparams(
            source=model_name, savedir=f"{data_path}/pretrained_models/{model_name}"
        )

    def test_step(self, batch, batch_idx):
        return ...

    def sample(self, x_start: Tensor, num_steps: int = 0, **kwargs) -> Tensor:
        target_device = next(self.model.parameters()).device
        self.model.device = str(target_device)
        x_start = x_start.to(target_device).squeeze(1)
        out = self.model.separate_batch(x_start)
        out = out.squeeze(-1).unsqueeze(1)  # shape (B, 1, T)
        # normalize to [-1, 1]
        out = out / out.abs().max(dim=-1, keepdim=True)[0]
        return out
