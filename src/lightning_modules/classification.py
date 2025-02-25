from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.nn import Module
from src.networks import BaseEncoderDecoder
from torch.optim import Optimizer
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from pytorch_lightning.utilities import grad_norm
from torch import Tensor

class Classifier(BaseLightningModule):
    def __init__(
        self, 
        model : Module,
        encoder_decoder : BaseEncoderDecoder | None = None,
        optimizer : Optimizer | None = None,
        lr_scheduler : dict[str, str] | None = None,
    ):
        super().__init__()
        self.model = model
        self.encoder_decoder = encoder_decoder
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        
    def forward(self, x : Tensor) -> Tensor:
        return self.model(x)
    
    def common_step(self, batch : tuple[Tensor, Tensor]) -> Tensor:
        x0, x1 = batch
        batch_size = x0.size(0)
        x0, x1 = self.encoder_decoder.encode(x0), self.encoder_decoder.encode(x1)
        
        zeros = torch.zeros(batch_size, 1, device=x0.device)
        ones = torch.ones(batch_size, 1, device=x0.device)
        
        input = torch.cat([x0, x1], dim=0)
        target = torch.cat([zeros, ones], dim=0)
        
        output = self.forward(input)
        loss = binary_cross_entropy_with_logits(output, target)
        accuracy = (output > 0).float().eq(target).float().mean()
    
        return {
            "loss": loss,
            "accuracy": accuracy
        }
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = grad_norm(self.model, norm_type=2)
        self.log_dict(grad_norms)
    
    def training_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        loss_dict = self.common_step(batch)
        loss_dict = {f"train_{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss_dict)
        return loss_dict["train_loss"]
    
    def validation_step(self, batch : tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        loss_dict = self.common_step(batch)
        loss_dict = {f"val_{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss_dict)
        return loss_dict["val_loss"]
    
    @torch.no_grad()
    def predict(self, x : Tensor) -> Tensor:
        self.eval()
        output = self(x)
        prediction = (output > 0).float()
        return prediction
        
    def configure_optimizers(self):
        optim = self.partial_optimizer(self.model.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }