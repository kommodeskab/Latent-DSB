from .baseloss import BaseLossFunction
from src import Batch, ModelOutput, LossOutput
from torch import Tensor
import torch


class DriftingLoss(BaseLossFunction):
    def __init__(
        self,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

    def compute_V(self, x: Tensor, y_pos: Tensor, y_neg: Tensor) -> Tensor:
        original_shape = x.shape
        batch_size = x.size(0)
        x = x.flatten(start_dim=1)
        y_pos = y_pos.flatten(start_dim=1)
        y_neg = y_neg.flatten(start_dim=1)

        n_pos = y_pos.size(0)

        dist_pos = torch.cdist(x, y_pos)
        dist_neg = torch.cdist(x, y_neg)

        dist_neg = dist_neg + torch.eye(batch_size, device=x.device) * 1e6

        logits_pos = -dist_pos / self.temperature
        logits_neg = -dist_neg / self.temperature

        logits = torch.cat([logits_pos, logits_neg], dim=1)

        A_row = logits.softmax(dim=-1)
        A_col = logits.softmax(dim=0)
        A = torch.sqrt(A_row * A_col)

        A_pos, A_neg = A[:, :n_pos], A[:, n_pos:]
        W_pos = A_pos.clone()
        W_neg = A_neg.clone()

        W_pos = W_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = W_neg * A_pos.sum(dim=1, keepdim=True)

        drift_pos = W_pos @ y_pos
        drift_neg = W_neg @ y_neg

        V = drift_pos - drift_neg

        return V.view(original_shape)

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        y_pos = batch["target"]
        x = model_output["output"]
        y_neg = x.detach()
        V = self.compute_V(x, y_pos, y_neg)
        target = (x + V).detach()
        loss = torch.nn.functional.mse_loss(x, target)
        return LossOutput(loss=loss)
