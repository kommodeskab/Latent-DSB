from speechbrain.inference.separation import SepformerSeparation
import pytorch_lightning as pl
from torch import Tensor
from dotenv import load_dotenv
import os

load_dotenv()
data_path = os.getenv("DATA_PATH")

class Sepformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_name = "speechbrain/sepformer-wham16k-enhancement"
        self.model: SepformerSeparation = SepformerSeparation.from_hparams(source=model_name, savedir=f"{data_path}/pretrained_models/{model_name}")
        
    def test_step(self, batch, batch_idx):
        return ...
    
    def sample(self, x_start: Tensor, num_steps: int, **kwargs) -> Tensor:
        x_start = x_start.squeeze(1)
        out = self.model.separate_batch(x_start)
        out = out.squeeze(-1).unsqueeze(1) # shape (B, 1, T)
        # normalize to [-1, 1]
        out = out / out.abs().max(dim=-1, keepdim=True)[0]
        return out
