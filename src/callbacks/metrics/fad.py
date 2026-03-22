from src.callbacks.metrics import BaseMetric
from transformers import ClapModel, ClapProcessor
import torch
from torch import Tensor
from torchaudio.functional import resample
from src import StepOutput, TensorDict, UnpairedAudioBatch
from src.lightning_modules import BaseLightningModule
from typing import Optional


class ClapEmbedder:
    def __init__(self):
        self.model: ClapModel = ClapModel.from_pretrained("laion/clap-htsat-fused")
        self.processor: ClapProcessor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        self.target_sample_rate = 48_000  # this clap model uses 48khz audio
        self.model.eval()

    def to(self, device: torch.device):
        self.model = self.model.to(device)

    def __call__(self, audio: Tensor, sample_rate: int) -> Tensor:
        if sample_rate != self.target_sample_rate:
            audio = resample(audio, orig_freq=sample_rate, new_freq=self.target_sample_rate)

        # shape (B, C, T) -> (B, T)
        audio = audio.mean(dim=1)

        # Convert to list of numpy arrays to ensure batch processing in transformer processor
        # If passed as a single tensor/array, it might be interpreted as a single multi-channel audio
        audio_input = list(audio.cpu().numpy())

        with torch.no_grad():
            # Pass sampling_rate specifically to avoid warning/ambiguity
            inputs = self.processor(audio=audio_input, return_tensors="pt", sampling_rate=self.target_sample_rate).to(
                self.model.device
            )
            outputs = self.model.get_audio_features(**inputs)

        return outputs.pooler_output


class FAD(BaseMetric):
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
        x = torch.cat(self.generated_embeddings, dim=0)
        y = torch.cat(self.real_embeddings, dim=0)

        mu_x = x.mean(dim=0)
        mu_y = y.mean(dim=0)

        sigma_x = torch.cov(x.T)
        sigma_y = torch.cov(y.T)

        a = (mu_x - mu_y).square().sum()
        b = sigma_x.trace() + sigma_y.trace()
        c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
        fad = a + b - 2 * c

        return {"value": fad}

    def reset(self):
        self.generated_embeddings = []
        self.real_embeddings = []

    def name(self) -> str:
        return f"FAD between '{self.output_key}' and '{self.real_key}'"
