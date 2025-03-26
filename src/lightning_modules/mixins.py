from src.networks import BaseEncoderDecoder
from torch import Tensor
import torch

class EncoderDecoderMixin:
    encoder_decoder : BaseEncoderDecoder
    
    @torch.no_grad()
    def encode(self, x : Tensor) -> Tensor:
        self.encoder_decoder.eval()
        x = self.encoder_decoder.encode(x)
        return x
    
    @torch.no_grad()
    def decode(self, x : Tensor) -> Tensor:
        self.encoder_decoder.eval()
        x = self.encoder_decoder.decode(x)
        return x