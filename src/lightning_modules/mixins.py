from src.networks import BaseEncoderDecoder
from torch import Tensor
import torch

class EncoderDecoderMixin:
    encoder_decoder : BaseEncoderDecoder
    
    def encode(self, x : Tensor) -> Tensor:
        self.encoder_decoder.eval()
        with torch.no_grad():
            x = self.encoder_decoder.encode(x)
        return x
    
    def decode(self, x : Tensor) -> Tensor:
        self.encoder_decoder.eval()
        with torch.no_grad():
            x = self.encoder_decoder.decode(x)
        return x