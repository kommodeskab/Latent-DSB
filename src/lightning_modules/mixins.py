from src.networks import BaseEncoderDecoder
from torch import Tensor
import torch

class EncoderDecoderMixin:
    added_noise : float
    encoder_decoder : BaseEncoderDecoder
    
    @torch.no_grad()
    def encode(self, x : Tensor, add_noise : bool = False) -> Tensor:
        self.encoder_decoder.eval()
        x = self.encoder_decoder.encode(x)
        if add_noise: 
            x = x + self.added_noise * torch.randn_like(x)
        return x
    
    @torch.no_grad()
    def decode(self, x : Tensor) -> Tensor:
        self.encoder_decoder.eval()
        x = self.encoder_decoder.decode(x)
        return x