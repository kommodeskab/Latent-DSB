import torch
import torch.nn as nn
import math

def sinusoidal_encoding(tensor, enc_size, exponential_base = 10000.0):
    device = tensor.device
    batch_size = tensor.size(0)

    position = torch.arange(0, enc_size, dtype=torch.float, device = device).unsqueeze(0)
    position = position.repeat(batch_size, 1)

    div_term = torch.exp(torch.arange(0, enc_size, 2, device = device).float() * (-math.log(exponential_base) / enc_size))

    position[:, 0::2] = torch.sin(tensor * div_term)
    position[:, 1::2] = torch.cos(tensor * div_term)

    return position

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        layer_sizes: list[int] = [64, 64],
    ):
        super().__init__()
        # out_features = layer_sizes[-1]

        self.layers = nn.ModuleList()
        for out_features in layer_sizes:
            self.layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x

class SimpleNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_time_embeddings : int,
        encoder_layers: list[int] = [16, 32],
        decoder_layers: list[int] = [32, 64, 128],
        time_encoding_size: int = 16,
    ):
        super().__init__()
        assert encoder_layers[-1] == decoder_layers[0], "Encoder and decoder must have the same middle size"
        middle_size = encoder_layers[-1]
        self.time_encoding_size = time_encoding_size
        self.time_embeddder = nn.Embedding(num_time_embeddings, time_encoding_size)

        self.x_encoder = MLP(in_features, encoder_layers)
        self.time_encoder = MLP(time_encoding_size, encoder_layers)
        self.decoder = MLP(2 * middle_size, decoder_layers + [out_features])

    def forward(self, x : torch.Tensor, t : torch.Tensor):
        time_embed = self.time_embeddder(t.long())

        x_enc = self.x_encoder(x)
        time_enc = self.time_encoder(time_embed)

        out = torch.cat([x_enc, time_enc], dim = 1)
        out = self.decoder(out)

        return out

class SimpleNetworkImages(SimpleNetwork):
    def __init__(
        self,
        channels : int,
        height : int, 
        width : int, 
        num_time_embeddings : int,
        time_encoding_size: int = 16,
    ):
        features = height * width * channels
        layers = [32, 32]
        super().__init__(features, features, num_time_embeddings, layers, layers, time_encoding_size)
        
    def forward(self, x : torch.Tensor, t : torch.Tensor):
        shape = x.size()
        x = x.view(shape[0], -1)
        x = super().forward(x, t)
        return x.view(shape)
    
if __name__ == "__main__":
    x = torch.randn(32, 1, 16, 16)
    t = torch.randint(0, 100, (32, ))
    model = SimpleNetworkImages(1, 16, 16, 100)
    out = model(x, t)
    print(out.size())
