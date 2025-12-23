from src.losses import BaseLoss
import torch
from torch.distributions import kl_divergence, Normal
from typing import Optional
import pytorch_lightning as pl

class VAELoss(BaseLoss):
    def __init__(
        self,
        beta: float = 1.0,
        target_kl: Optional[float]= None,
    ):
        super().__init__()
        self.beta = beta
        self.update_rate = 1.001
        self.target_kl = target_kl
        
    def update_beta(self, kl_divergence: float) -> None:
        update_rate = self.update_rate if kl_divergence > self.target_kl else 1 / self.update_rate
        self.beta = max(self.beta * update_rate, 1e-5)
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        q_z: Normal = batch["q_z"]
        p_x: Normal = batch["p_x"]
        x: torch.Tensor = batch["target"]
        model: pl.LightningModule = batch["model"]
        
        B, *shape = x.shape
        x_size = torch.tensor(shape).prod()
        
        standard_normal = Normal(0, 1)
        kl_loss = kl_divergence(q_z, standard_normal).reshape(B, -1).sum(dim=1)
        kl_loss = kl_loss.mean() / x_size
        
        if model.training and self.target_kl is not None:
            self.update_beta(kl_loss.item())
        
        recon_loss = -p_x.log_prob(x).reshape(B, -1).sum(dim=1)
        recon_loss = recon_loss.mean() / x_size
        
        loss = recon_loss + self.beta * kl_loss
        
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "z_std": q_z.scale.mean(),
            "x_std": p_x.scale.mean(),
            "beta": torch.tensor(self.beta),
        }