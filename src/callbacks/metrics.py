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
    def __init__(self, sample_rate : int, device : str):
        # initialize models for transcription
        self.device = device
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000 # Whisper model expects 16kHz
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
        self.model.config.forced_decoder_ids = None
        self.model.eval()
        self.wer_metric = WordErrorRate()
        
        self.real_transcriptions = []
        self.generated_transcriptions = []
    
    def resample(self,  audio : Tensor) -> Tensor:
        if self.sample_rate != self.target_sample_rate:
            resampler = Resample(self.sample_rate, self.target_sample_rate)
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
            predicted_ids = self.model.generate(input_features, language="en")
        transcription : str = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcription = self.normalize_text(transcription)
        return transcription
    
    def batch_transcribe(self, audios : Tensor):
        transcriptions = []
        for audio in audios:
            transcription = self.transcribe(audio)
            transcriptions.append(transcription)
        return transcriptions
    
    def update(self, generated : Tensor, real : Tensor) -> None:
        real = self.resample(real)
        generated = self.resample(generated)
        real_transcriptions = self.batch_transcribe(real)
        generated_transcriptions = self.batch_transcribe(generated)
        self.real_transcriptions.extend(real_transcriptions)
        self.generated_transcriptions.extend(generated_transcriptions)
        
    def compute(self) -> float:
        wer = self.wer_metric(self.generated_transcriptions, self.real_transcriptions)
        return wer
        
    
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
    
class SRCS:
    def __init__(self, sample_rate : int, device : str):
        assert sample_rate == 16000, "SRCS model only supports 16kHz sample rate"
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        self.model : EncDecSpeakerLabelModel = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        self.model = self.model.to(device)
        self.model.freeze()
        self.avg_cos_sims = []
    
    @torch.no_grad()
    def get_embedding(self, audio : Tensor) -> Tensor:
        audio_len = torch.tensor([audio.shape[1]] * audio.shape[0]).to(audio.device)
        _, emb = self.model.forward(input_signal=audio, input_signal_length=audio_len)
        return emb
    
    def update(self, generated : Tensor, real : Tensor):
        assert real.shape == generated.shape
        real_emb = self.get_embedding(real) # shape = (batch_size, 192)
        generated_emb = self.get_embedding(generated) # shape = (batch_size, 192)
        # find the average cosine similarity between the paired embeddings
        avg_cosine_similarity = torch.nn.functional.cosine_similarity(real_emb, generated_emb).mean().item()
        self.avg_cos_sims.append(avg_cosine_similarity)
        
    def compute(self) -> float:
        return sum(self.avg_cos_sims) / len(self.avg_cos_sims)
    
def calculate_curvature_displacement(trajectories : Tensor, timeschedule : Tensor) -> Tensor:
    # trajectories.shape = (trajectory_length, batch_size, ...)
    traj_len, batch_size, *shape = trajectories.shape
    trajectories = trajectories.permute(1, 0, *range(2, len(shape) + 2))
    trajectories = trajectories.reshape(batch_size, traj_len, -1)
    
    C_ts = []
    for trajectory in trajectories:
        x0, x1 = trajectory[0], trajectory[-1]
        dx = trajectory[1:] - trajectory[:-1]
        dt = timeschedule[1:] - timeschedule[:-1]
        dt = dt.unsqueeze(-1)
        C_t = (x1 - x0) - dx / dt
        C_t = C_t.norm(dim = 1) / (x1 - x0).norm(dim = 0)
        C_ts.append(C_t)
        
    C_ts = torch.stack(C_ts, dim=0)
    return C_ts

if __name__ == "__main__":
    x1 = torch.randn(16, 1, 16000)
    x2 = torch.randn(16, 1, 16000)
    metric = PESQ()
    print(metric.evaluate(x1, x2))