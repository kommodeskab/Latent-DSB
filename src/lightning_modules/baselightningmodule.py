import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch.nn.init as init
import torch

class BaseLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def _convert_dict_losses(self, losses, suffix = "", prefix = ""):
        if suffix:
            losses = {f"{k}/{suffix}": v for k, v in losses.items()}
        if prefix:
            losses = {f"{prefix}/{k}": v for k, v in losses.items()}
        return losses
    
    @property
    def logger(self) -> WandbLogger:
        return self.trainer.logger
    
    @staticmethod
    def init_weights(model : nn.Module) -> None:
        """
        Initializes the weights of the forward and backward models  
        using the Kaiming Normal initialization
        """
        @torch.no_grad()
        def initialize(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Apply initialization to both networks
        model.apply(initialize)