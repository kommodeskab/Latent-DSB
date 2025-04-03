from src.lightning_modules.baselightningmodule import BaseLightningModule
from diffusers import AutoencoderKL
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_ema import ExponentialMovingAverage
from pytorch_lightning.utilities import grad_norm
from torch import Tensor

class VAE(BaseLightningModule):
    def __init__(
        self,
        model : AutoencoderKL,
        optimizer : Optimizer,
        lr_scheduler : LRScheduler,
        beta : float = 0.1,
        ema_decay : float = 0.999,
        n_fft : int = 1024,
        hop_length : int = 256,
    ):
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        self.beta = beta
        self.n_fft = n_fft
        self.hop_length = hop_length
        super().__init__()
        
    def on_fit_start(self):
        self.ema.to(self.device)
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)
        
    def make_stft(self, audio : Tensor) -> Tensor:
        
        
    def common_step(self, audio : Tensor) -> Tensor:
        