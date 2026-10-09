import math
import torch
from torch import Tensor
from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput


class DriftingLoss(BaseLossFunction):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError(f"temperature must be a positive float, got {temperature}.")
        self.temperature = float(temperature)

    def compute_V(self, x: Tensor, y_pos: Tensor, y_neg: Tensor) -> Tensor:
        original_shape = x.shape
        batch_size = x.size(0)

        x_flat = x.flatten(start_dim=1)
        y_pos_flat = y_pos.flatten(start_dim=1)
        y_neg_flat = y_neg.flatten(start_dim=1)

        n_pos = y_pos_flat.size(0)
        dim = x_flat.size(1)

        # Pairwise normalized Euclidean distance (normalized by sqrt(dim) as in Deng et al., 2026)
        dist_pos = torch.cdist(x_flat, y_pos_flat) / math.sqrt(dim)
        dist_neg = torch.cdist(x_flat, y_neg_flat) / math.sqrt(dim)

        # Ignore self-repulsion if y_neg corresponds to generated samples x
        if x is y_neg or (batch_size == y_neg_flat.size(0) and torch.equal(x_flat, y_neg_flat)):
            dist_neg = dist_neg + torch.eye(batch_size, device=x.device, dtype=dist_neg.dtype) * 1e6

        temp = self.temperature

        logits_pos = -dist_pos / temp
        logits_neg = -dist_neg / temp
        logits = torch.cat([logits_pos, logits_neg], dim=1)

        # Doubly normalized affinity matrix (Deng et al., Algorithm 2)
        A_row = logits.softmax(dim=-1)
        A_col = logits.softmax(dim=0)
        A = torch.sqrt(A_row * A_col)

        A_pos, A_neg = A[:, :n_pos], A[:, n_pos:]

        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

        # Kernel-gradient drift for Laplace kernel (Turan et al., 2026)
        # Avoid division by zero when dist <= eps where displacement is 0
        eps = 1e-7
        mask_pos = dist_pos > eps
        W_pos_div = torch.zeros_like(W_pos)
        W_pos_div[mask_pos] = W_pos[mask_pos] / dist_pos[mask_pos]

        mask_neg = dist_neg > eps
        W_neg_div = torch.zeros_like(W_neg)
        W_neg_div[mask_neg] = W_neg[mask_neg] / dist_neg[mask_neg]

        drift_pos = (W_pos_div @ y_pos_flat) - x_flat * W_pos_div.sum(dim=1, keepdim=True)
        drift_neg = (W_neg_div @ y_neg_flat) - x_flat * W_neg_div.sum(dim=1, keepdim=True)

        V = (drift_pos - drift_neg) * (1.0 / temp)

        return V.view(original_shape)

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        x = model_output["output"]
        y_pos = batch["target"]
        y_neg = x

        with torch.no_grad():
            V = self.compute_V(x, y_pos, y_neg)
            target = x + V

        loss = torch.nn.functional.mse_loss(x, target)

        return LossOutput(loss=loss)
