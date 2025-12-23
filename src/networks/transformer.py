import torch
import torch.nn as nn
import math

class ConditionalTimeSeriesTransformer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int,
        d_model: int, 
        nhead: int, 
        num_layers: int, 
        causal: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.causal = causal
        
        self.class_embedding = nn.Embedding(num_classes, d_model)
        
        time_emb_dim = d_model // 2 
        self.timestep_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        combined_dim = input_dim + d_model + time_emb_dim
        
        self.input_projection = nn.Linear(combined_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4*d_model,
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(
        self, 
        src: torch.Tensor, 
        timesteps: torch.Tensor, 
        conditional: torch.Tensor
    ) -> torch.Tensor:
        
        src = src.permute(0, 2, 1)
        B, L, C = src.shape

        c_emb = self.class_embedding(conditional).unsqueeze(1).expand(-1, L, -1)
        
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(B, L)
        t_emb = self.timestep_mlp(timesteps.unsqueeze(-1))
        
        combined_input = torch.cat((src, c_emb, t_emb), dim=2) # shape (B, L, C + d_model + time_emb_dim)
        
        x = self.input_projection(combined_input) # (B, L, d_model)
        x = self.dropout(x)
        
        x = self.pos_encoder(x)
        
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
            
        output = self.transformer_encoder(x, mask=mask, is_causal=self.causal)
        
        prediction = self.output_projection(output)  # (B, L, C)
        prediction = prediction.permute(0, 2, 1)    # (B, C, L)
        
        return prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)