import torch
from torch import Tensor
from torchmetrics.functional.text import word_error_rate
import re
from .base import BaseMetric
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Optional
from src.lightning_modules import BaseLightningModule
from src import StepOutput, TensorDict, UnpairedAudioBatch
from torchaudio.functional import resample


class WER(BaseMetric):
    def __init__(self, clean_key: str, output_key: str, log_transcriptions: bool = False):
        self.clean_key = clean_key
        self.output_key = output_key
        self.log_transcriptions = log_transcriptions

        # initialize models for transcription
        self.target_sample_rate = 16000  # Whisper model expects 16kHz
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
        )
        self.model.config.forced_decoder_ids = None
        self.model.eval()
        self.real_transcriptions = []
        self.generated_transcriptions = []

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    @staticmethod
    def normalize_text(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
        s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
        return s.strip()

    def transcribe(self, audios: Tensor):
        audios = audios.squeeze(1)  # shape (batch_size, length)
        transcriptions = []

        for audio in audios:
            # audio is a tensor of shape (length,)
            processed = self.processor(audio.cpu(), sampling_rate=16000, return_tensors="pt")
            input_features = processed.input_features.to(audios.device)
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, language="en")
            transcription = self.processor.decode(predicted_ids, skip_special_tokens=True)[0]
            transcription = self.normalize_text(transcription)
            transcriptions.append(transcription)

        return transcriptions

    def add(
        self,
        pl_module: BaseLightningModule,
        outputs: StepOutput,
        batch: UnpairedAudioBatch,
        batch_idx: int,
        extras: Optional[TensorDict] = None,
    ):
        real = batch[self.clean_key]
        generated = extras[self.output_key]
        sample_rate = batch["sample_rate"][0]

        if sample_rate != self.target_sample_rate:
            real = resample(real, sample_rate, self.target_sample_rate)
            generated = resample(generated, sample_rate, self.target_sample_rate)

        real_transcriptions = self.transcribe(real)
        generated_transcriptions = self.transcribe(generated)

        self.real_transcriptions.extend(real_transcriptions)
        self.generated_transcriptions.extend(generated_transcriptions)

        self.module = pl_module

    def compute(self) -> float:
        wer = word_error_rate(preds=self.generated_transcriptions, target=self.real_transcriptions)

        if self.log_transcriptions:
            self.module.logger.log_text(
                f"WER Transcriptions ({self.name()})",
                columns=["Real Transcription", "Generated Transcription"],
                data=list(zip(self.real_transcriptions, self.generated_transcriptions)),
            )

        return wer.item()

    def reset(self) -> None:
        self.real_transcriptions = []
        self.generated_transcriptions = []

    def name(self) -> str:
        return f"WER clean='{self.clean_key}' prediction='{self.output_key}'"
