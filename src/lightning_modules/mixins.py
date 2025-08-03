from src.networks import BaseEncoderDecoder
from torch import Tensor
import torch

class EncoderDecoderMixin:
    encoder_decoder : BaseEncoderDecoder
    
    @torch.no_grad()
    def encode(self, x : Tensor) -> Tensor:
        if self.encoder_decoder.training:
            self.encoder_decoder.eval()
        x = self.encoder_decoder.encode(x)
        return x

    @torch.no_grad()
    def decode(self, x : Tensor) -> Tensor:
        if self.encoder_decoder.training:
            self.encoder_decoder.eval()
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
    
    def encode_batch(self, *xs : Tensor) -> tuple[Tensor, ...]:
        """
        Encodes a batch of tensors. The tensors are expected to be in the same format as the encoder's input.
        """
        assert all(x.shape == xs[0].shape for x in xs), "All tensors must have the same shape"
        xs_stacked = torch.cat(xs, dim=0)
        xs_encoded = self.encode(xs_stacked)
        return xs_encoded.chunk(len(xs), dim=0) 