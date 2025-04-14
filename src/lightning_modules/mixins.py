from src.networks import BaseEncoderDecoder
from torch import Tensor
import torch

class EncoderDecoderMixin:
    encoder_decoder : BaseEncoderDecoder
    
    def encode(self, x : Tensor) -> Tensor:
        if self.encoder_decoder.training:
            self.encoder_decoder.eval()
        with torch.no_grad():
            x = self.encoder_decoder.encode(x)
        return x
    
    def decode(self, x : Tensor) -> Tensor:
        if self.encoder_decoder.training:
            self.encoder_decoder.eval()
        with torch.no_grad():
            x = self.encoder_decoder.decode(x)
        return x
    
    def encode_batch(self, batch : Tensor) -> Tensor:
        x0, x1 = batch
        xs_stacked = torch.cat([x0, x1], dim=0)
        xs_encoded = self.encode(xs_stacked)
        x0_encoded, x1_encoded = xs_encoded.chunk(2, dim=0)
        return x0_encoded, x1_encoded