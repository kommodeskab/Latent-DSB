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
    
    def chunk_encode(self, x : Tensor, chunk_size : int) -> Tensor:
        chunked_x = torch.split(x, chunk_size, dim=0)
        encoded_chunks = [self.encode(chunk) for chunk in chunked_x]
        return torch.cat(encoded_chunks, dim=0)
    
    def chunk_decode(self, x : Tensor, chunk_size : int) -> Tensor:
        chunked_x = torch.split(x, chunk_size, dim=0)
        decoded_chunks = [self.decode(chunk) for chunk in chunked_x]
        return torch.cat(decoded_chunks, dim=0)
    
    
    def encode_batch(self, batch : Tensor) -> Tensor:
        x0, x1 = batch
        xs_stacked = torch.cat([x0, x1], dim=0)
        xs_encoded = self.encode(xs_stacked)
        x0_encoded, x1_encoded = xs_encoded.chunk(2, dim=0)
        return x0_encoded, x1_encoded