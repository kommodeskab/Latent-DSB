import math
import torch
from torch import Tensor
from typing import Optional, Literal
from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput


class DriftingLoss(BaseLossFunction):
    def __init__(
        self,
        temperature: Optional[float] = None,
        kernel_type: Literal["gaussian", "laplace"] = "gaussian",
        log: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        if kernel_type not in ("gaussian", "laplace"):
            raise ValueError(f"Invalid kernel_type: {kernel_type}. Expected 'gaussian' or 'laplace'.")
        self.kernel_type = kernel_type
        self.log = log

    def compute_V(self, x: Tensor, y_pos: Tensor, y_neg: Tensor) -> Tensor:
        original_shape = x.shape
        batch_size = x.size(0)

        x_flat = x.flatten(start_dim=1)
        y_pos_flat = y_pos.flatten(start_dim=1)
        y_neg_flat = y_neg.flatten(start_dim=1)

        n_pos = y_pos_flat.size(0)
        dim = x_flat.size(1)

        dist_pos = torch.cdist(x_flat, y_pos_flat) / math.sqrt(dim)
        dist_neg = torch.cdist(x_flat, y_neg_flat) / math.sqrt(dim)

        dist_neg = dist_neg + torch.eye(batch_size, device=x.device) * 1e6

        # normalize the distances with the dimension of the space
        if self.kernel_type == "gaussian":
            kernel_dist_pos = dist_pos.pow(2)
            kernel_dist_neg = dist_neg.pow(2)
        else:
            kernel_dist_pos = dist_pos
            kernel_dist_neg = dist_neg

        if self.temperature is None:
            with torch.no_grad():
                all_dists = torch.cat([kernel_dist_pos.flatten(), kernel_dist_neg.flatten()])
                valid_dists = all_dists[all_dists < 1e5]
                temp = torch.median(valid_dists).item() if valid_dists.numel() > 0 else 1.0
                temp = max(temp, 1e-4)
        else:
            temp = float(self.temperature)

        logits_pos = -kernel_dist_pos / temp
        logits_neg = -kernel_dist_neg / temp
        logits = torch.cat([logits_pos, logits_neg], dim=1)

        A_row = logits.softmax(dim=-1)
        A_col = logits.softmax(dim=0)
        A = torch.sqrt(A_row * A_col)

        A_pos, A_neg = A[:, :n_pos], A[:, n_pos:]

        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

        if self.kernel_type == "gaussian":
            drift_pos = W_pos @ y_pos_flat
            drift_neg = W_neg @ y_neg_flat

            V = (drift_pos - drift_neg) * (2.0 / temp)
        else:
            drift_pos = (W_pos @ y_pos_flat) - x_flat * W_pos.sum(dim=1, keepdim=True)
            drift_neg = (W_neg @ y_neg_flat) - x_flat * W_neg.sum(dim=1, keepdim=True)

            V = (drift_pos - drift_neg) * (1.0 / temp)

        return V.view(original_shape)

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        x = model_output["output"]
        y_pos = batch["target"]
        y_neg = x.detach()

        V = self.compute_V(x, y_pos, y_neg)
        target = (x + V).detach()

        loss = torch.nn.functional.mse_loss(x, target)

        if self.log:
            loss = 10 * torch.log10(loss + 1e-6)  # Convert to dB scale, adding a small value to avoid log(0)

        return LossOutput(loss=loss)
