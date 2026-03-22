from .base import BaseMetric
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch import Tensor
import logging
import torch
from torchaudio.functional import resample
from torch.nn.functional import cosine_similarity
from src.lightning_modules import BaseLightningModule
from src import StepOutput, TensorDict, UnpairedAudioBatch
from typing import Optional
import warnings

# ignore warnings from Nemo
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"nemo\.collections\.asr\.parts\.utils\.vad_utils")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)


class SRCS(BaseMetric):
    """
    Speaker Recognition Cosine Similarity (SRCS) metric for unpaired audio translation.
    This metric calculates the cosine similarity between the speaker embeddings of the clean and output audio samples,
    to validate whether the model is able to preserve the speaker identity during the translation process.
    It uses a pre-trained speaker verification model from NVIDIA's NeMo toolkit to extract speaker embeddings.
    """

    def __init__(
        self,
        clean_key: str,
        output_key: str,
    ):
        self.target_sample_rate = 16000
        self.model: EncDecSpeakerLabelModel = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.model.freeze()
        self.values = []
        self.output_key = output_key
        self.clean_key = clean_key

    def to(self, device: str):
        self.model.to(device)

    @torch.no_grad()
    def get_embedding(self, audio: Tensor, sample_rate: int) -> Tensor:
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        audio = resample(audio, orig_freq=sample_rate, new_freq=self.target_sample_rate)
        audio_len = torch.tensor([audio.shape[1]] * audio.shape[0]).to(audio.device)
        _, emb = self.model.forward(input_signal=audio, input_signal_length=audio_len)
        return emb

    def add(
        self,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: UnpairedAudioBatch,
        batch_idx: int,
        extras: Optional[TensorDict] = None,
    ):
        sample_rate = batch["sample_rate"][0]

        output = extras[self.output_key]
        clean = batch[self.clean_key]

        output_emb = self.get_embedding(output, sample_rate)
        clean_emb = self.get_embedding(clean, sample_rate)

        cos_sim = cosine_similarity(output_emb, clean_emb).tolist()
        self.values.extend(cos_sim)

    def compute(self) -> TensorDict:
        values = torch.tensor(self.values)
        return {
            "mean": values.mean(),
            "std": values.std(),
        }

    def reset(self) -> None:
        self.values = []

    def name(self) -> str:
        return f"SRCS clean='{self.clean_key}' output='{self.output_key}'"
