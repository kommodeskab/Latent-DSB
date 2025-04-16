import distillmos
import torch
from torch import Tensor
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torchmetrics.text import WordErrorRate
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchaudio.transforms import Resample
import re


class MOS:
    def __init__(self, device = 'cuda'):
        sqa_model = distillmos.ConvTransformerSQAModel().to(device)
        sqa_model.eval()
        self.sqa_model = sqa_model
        self.device = device
    
    def evaluate(self, samples : Tensor, old_sample_rate : int) -> float:
        samples = samples.to(self.device)
        if samples.dim() == 3:
            samples = samples.mean(dim=1)
        # resample to 16kHz (this is what the model expects)
        if old_sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(old_sample_rate, 16000).to(self.device)
            samples = resampler(samples)
            
        with torch.no_grad():
            mos : Tensor = self.sqa_model(samples)
        mos = mos.mean()
        return mos
    
class KAD:
    def __init__(self, alpha : float = 100.):
        self.alpha = alpha
    
    @staticmethod
    def kernel(x : Tensor, y : Tensor) -> Tensor:
        return torch.exp(-torch.norm(x - y) ** 2)
    
    @torch.no_grad()
    def evaluate(self, generated : Tensor, real : Tensor) -> float:
        assert generated.shape[1:] == real.shape[1:], "The generated and real tensors must have the same shape."
        assert generated.device == real.device, "The generated and real tensors must be on the same device."
        
        n = generated.size(0)
        m = real.size(0)
        
        Kxx = torch.zeros(n, n)
        Kyy = torch.zeros(m, m)
        Kxy = torch.zeros(n, m)
        
        for i in range(n):
            for j in range(n):
                Kxx[i, j] = self.kernel(generated[i], generated[j])
                
        for i in range(m):
            for j in range(m):
                Kyy[i, j] = self.kernel(real[i], real[j])
                
        for i in range(n):
            for j in range(m):
                Kxy[i, j] = self.kernel(generated[i], real[j])
                
        Kxx = Kxx.sum() / (n * (n - 1))
        Kyy = Kyy.sum() / (m * (m - 1))
        Kxy = Kxy.sum() / (n * m)
        
        return self.alpha * (Kxx + Kyy - 2 * Kxy)
class WER:
    def __init__(self, original_audios : Tensor, original_sample_rate : int):
        # initialize models for transcription
        self.device = original_audios.device
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
        self.model.eval()
        
        if original_sample_rate != 16000:
            original_audios = self.resample(original_audios, original_sample_rate, 16000)
        
        print("Transcribing original audios...")
        self.original_transcriptions = self.batch_transcribe(original_audios)
        self.wer_metric = WordErrorRate()
    
    @staticmethod
    def resample(audio : Tensor, original_sample_rate : int, target_sample_rate : int) -> Tensor:
        if original_sample_rate != target_sample_rate:
            resampler = Resample(original_sample_rate, target_sample_rate)
            audio = resampler(audio)
        return audio
    
    @staticmethod
    def normalize_text(s : str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
        s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
        return s.strip()
    
    def transcribe(self, audio : Tensor):
        # audio is a tensor of shape (1, length)
        processed = self.processor(audio.squeeze().cpu(), sampling_rate=16000, return_tensors="pt")
        input_features = processed.input_features.to(self.device)
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, language='en')
        transcription : str = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcription = self.normalize_text(transcription)
        return transcription
    
    def batch_transcribe(self, audios : Tensor):
        transcriptions = []
        for audio in audios:
            transcription = self.transcribe(audio)
            transcriptions.append(transcription)
        return transcriptions
    
    def compute_wer(self, audios : Tensor, sample_rate : int) -> tuple[Tensor, list[str]]:
        if sample_rate != 16000:
            audios = self.resample(audios, sample_rate, 16000)
        
        print("Transcribing generated audios and computing WER...")
        transcriptions = self.batch_transcribe(audios)
        wer = self.wer_metric(transcriptions, self.original_transcriptions)
        return wer, transcriptions
    
class SISDR:
    def __init__(self):
        self.sisdr = ScaleInvariantSignalDistortionRatio()
    
    def evaluate(self, generated : Tensor, real : Tensor) -> float:
        sisdr = self.sisdr(generated, real)
        return sisdr

class PESQ:
    def __init__(self, sample_rate : int = 16000):
        self.sample_rate = sample_rate
        mode = 'wb' if sample_rate >= 16000 else 'nb'
        self.pesq = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode=mode)
    
    def evaluate(self, generated : Tensor, real : Tensor) -> float:
        pesq = self.pesq(generated, real)
        return pesq    

if __name__ == "__main__":
    x1 = torch.randn(16, 1, 16000)
    x2 = torch.randn(16, 1, 16000)
    metric = PESQ()
    print(metric.evaluate(x1, x2))