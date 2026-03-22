from src.callbacks.metrics import BaseMetric
import torch
from src import StepOutput, TensorDict, UnpairedAudioBatch
from src.lightning_modules import BaseLightningModule
from typing import Optional
from src.callbacks.metrics.fad import ClapEmbedder

"""
KAD: Kernel Audio Distance - A perceptual metric for evaluating the similarity between generated and real audio samples using CLAP embeddings and MMD.
Inspired by: https://github.com/YoonjinXD/kadtk/blob/main/kadtk/kad.py
"""


class KAD(BaseMetric):
    def __init__(
        self,
        output_key: str,
        real_key: str,
    ):
        super().__init__()
        self.output_key = output_key
        self.real_key = real_key

        self.generated_embeddings = []
        self.real_embeddings = []

        self.clap_embedder = ClapEmbedder()

    def to(self, device: torch.device) -> None:
        self.clap_embedder.to(device)

    def add(
        self,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: UnpairedAudioBatch,
        batch_idx: int,
        extras: Optional[TensorDict] = None,
    ):
        generated = extras[self.output_key]
        real = batch[self.real_key]
        sr = batch["sample_rate"][0]

        generated_embeds = self.clap_embedder(generated, sr)
        real_embeds = self.clap_embedder(real, sr)

        self.generated_embeddings.append(generated_embeds.cpu())
        self.real_embeddings.append(real_embeds.cpu())

    def compute(self) -> TensorDict:
        x = torch.cat(self.real_embeddings, dim=0)  # (M, D) - Real
        y = torch.cat(self.generated_embeddings, dim=0)  # (N, D) - Generated

        m, n = x.size(0), y.size(0)

        # 2. Bandwidth Heuristic: Median distance of the reference (real) set
        # We calculate pairwise Euclidean distances for the real embeddings
        dists_real = torch.cdist(x, x, p=2)
        # Get only the upper triangle (excluding diagonal) to find median of pairwise distances
        mask = torch.triu(torch.ones(m, m), diagonal=1).bool()
        sigma = torch.median(dists_real[mask])

        # 3. Kernel Calculation
        # RBF Kernel: K(x, y) = exp(-||x-y||^2 / (2 * sigma^2))
        gamma = 1.0 / (2 * sigma**2 + 1e-8)

        def compute_kernel_matrix(t1, t2):
            # Efficient pairwise squared distance
            dist_sq = torch.cdist(t1, t2, p=2) ** 2
            return torch.exp(-gamma * dist_sq)

        k_xx = compute_kernel_matrix(x, x)
        k_yy = compute_kernel_matrix(y, y)
        k_xy = compute_kernel_matrix(x, y)

        # 4. Unbiased MMD^2 Estimator
        # Term 1: 1/(m*(m-1)) * sum_{i!=j} K(x_i, x_j)
        # Note: K(x_i, x_i) is always 1 for RBF
        t1 = (k_xx.sum() - m) / (m * (m - 1))

        # Term 2: 1/(n*(n-1)) * sum_{i!=j} K(y_i, y_j)
        t2 = (k_yy.sum() - n) / (n * (n - 1))

        # Term 3: 2/(m*n) * sum_{i,j} K(x_i, y_j)
        t3 = 2 * k_xy.mean()

        kad_score = 1000 * (t1 + t2 - t3)

        return {"value": kad_score}

    def reset(self):
        self.generated_embeddings = []
        self.real_embeddings = []

    def name(self) -> str:
        return f"KAD between '{self.output_key}' and '{self.real_key}'"
