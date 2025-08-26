import torch
from torch import Tensor
import torchaudio
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from torchmetrics.functional.text import word_error_rate
import re
from tqdm import tqdm
from transformers import ClapAudioModelWithProjection, ClapProcessor
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from torchaudio.functional import resample


class Metric:
    values : list
    
    @property
    def num_samples(self) -> int:
        return len(self.values)
    
    def update(self, *args, **kwargs) -> None: ...
    
    def compute(self) -> float:
        vals = torch.tensor(self.values)
        mean, std = vals.mean(), vals.std()
        return mean.item(), std.item()

class ResampleMixin:
    sample_rate : int
    target_sample_rate : int
    
    def resample(self,  audio : Tensor) -> Tensor:
        if self.sample_rate != self.target_sample_rate:
            audio = resample(audio, self.sample_rate, self.target_sample_rate)
        return audio
    
class SNR:
    def __init__(self):
        self.values = []
    
    def update(self, generated : Tensor, real : Tensor) -> None:
        if generated.dim() == 3:
            generated = generated.mean(dim=1)
        if real.dim() == 3:
            real = real.mean(dim=1)

        snr = scale_invariant_signal_distortion_ratio(generated, real).tolist()
        self.values.extend(snr)

class MOS:
    def __init__(self, device = 'cuda'):
        import distillmos
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
    
class KAD(ResampleMixin):
    def __init__(self, sample_rate : int, device : str = 'cuda'):
        super().__init__()
        self.alpha = 100
        self.sigma = 1
        
        self.sample_rate = sample_rate
        self.target_sample_rate = 48000
        
        self.generated_embeddings = []
        self.real_embeddings = []
        
        self.device = device
        self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").to(device)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        self.model.eval()
        
    def embed(self, audio : Tensor) -> Tensor:
        audio = self.resample(audio.cpu())
        with torch.no_grad():
            inputs = self.processor(audios=audio, return_tensors="pt", sampling_rate=48000).to(self.device)
            outputs = self.model(**inputs)
        return outputs.audio_embeds.cpu()
        
    def update(self, generated : Tensor, real : Tensor) -> None:
        generated_embeds = [self.embed(x) for x in generated]
        real_embeds = [self.embed(x) for x in real]
        
        for x, y in zip(generated_embeds, real_embeds):
            self.generated_embeddings.append(x)
            self.real_embeddings.append(y)
    
    def kernel(self, x : Tensor, y : Tensor) -> Tensor:
        norm = torch.norm(x - y)
        return torch.exp(-norm ** 2 / (2 * self.sigma ** 2))
    
    @torch.no_grad()
    def evaluate(self, generated : list, real : list) -> float:
        n = len(generated)
        m = len(real)
        
        generated = torch.stack(generated, dim=0).reshape(n, -1)
        real = torch.stack(real, dim=0).reshape(m, -1)
        
        k_xi_xj = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                kernel_output = self.kernel(generated[i], generated[j])
                k_xi_xj += kernel_output
                
        k_yi_yj = 0.0                
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                kernel_output = self.kernel(real[i], real[j])
                k_yi_yj += kernel_output

        k_xi_yj = 0.0                
        for i in range(n):
            for j in range(m):
                kernel_output = self.kernel(generated[i], real[j])
                k_xi_yj += kernel_output
                
        mmd = 1 / (n * (n - 1)) * k_xi_xj + 1 / (m * (m - 1)) * k_yi_yj - 2 / (n * m) * k_xi_yj
        return self.alpha * mmd.item()  # scale the MMD by alpha
    
    def compute(self) -> float:
        # for compatibility, we also return 0 as the stand-in standard deviation
        return self.evaluate(self.generated_embeddings, self.real_embeddings)

    
class WER(Metric, ResampleMixin):
    def __init__(self, sample_rate : int, device : str):
        # initialize models for transcription
        self.device = device
        self.sample_rate = sample_rate
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.target_sample_rate = 16000 # Whisper model expects 16kHz
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
        self.model.config.forced_decoder_ids = None
        self.model.eval()
        self.real_transcriptions = []
        self.generated_transcriptions = []
        self.values = []
    
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
        for audio in tqdm(audios, desc="Transcribing audio", leave=False, total=len(audios)):
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
        for real, generated in zip(real_transcriptions, generated_transcriptions):
            real, generated = [real], [generated]
            wer = word_error_rate(generated, real)
            self.values.append(wer.item())

class DNSMOS(Metric, ResampleMixin):
    def __init__(self, sample_rate : int = 16000, device : str = 'cuda'):
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000
        self.device = device
        self.values = []
    
    @torch.no_grad()
    def evaluate(self, samples : Tensor) -> Tensor:
        mos = deep_noise_suppression_mean_opinion_score(samples, fs=self.target_sample_rate, personalized=False, device=self.device)
        return mos[:, 3] # the 4th column is the mean opinion score
    
    def update(self, samples : Tensor) -> None:
        samples = self.resample(samples)
        mos = self.evaluate(samples)
        self.values.extend(mos.tolist())
        
class SRCS(Metric, ResampleMixin):
    def __init__(self, sample_rate : int, device : str):
        import logging
        logging.getLogger('nemo_logger').setLevel(logging.ERROR)
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        self.model : EncDecSpeakerLabelModel = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        self.model = self.model.to(device)
        self.model.freeze()
        self.values = []
    
    @torch.no_grad()
    def get_embedding(self, audio : Tensor) -> Tensor:
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        audio = self.resample(audio)
        audio_len = torch.tensor([audio.shape[1]] * audio.shape[0]).to(audio.device)
        _, emb = self.model.forward(input_signal=audio, input_signal_length=audio_len)
        return emb
    
    def update(self, generated : Tensor, real : Tensor):
        assert real.shape == generated.shape, f"Shapes of real and generated tensors must match: {real.shape} != {generated.shape}"
        real_emb = self.get_embedding(real) # shape = (batch_size, 192)
        generated_emb = self.get_embedding(generated) # shape = (batch_size, 192)
        # find the average cosine similarity between the paired embeddings
        cos_sim = torch.nn.functional.cosine_similarity(real_emb, generated_emb).tolist()
        self.values.extend(cos_sim)
    
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

def calculate_trajectory_length(trajectories : torch.Tensor, normalize : bool = True) -> torch.Tensor:
    # calculate the trajectory length and compare with ideal trajectory length (x1 - x0)
    # trajectories.shape = (num_steps, batch_size, ...)
    traj_len, batch_size, *shape = trajectories.shape
    trajectories = trajectories.permute(1, 0, *range(2, len(shape) + 2))
    trajectories = trajectories.reshape(batch_size, traj_len, -1)
    
    lengths = []
    
    for trajectory in trajectories:
        x0, x1 = trajectory[0], trajectory[-1]

        # sum the distances between consecutive points
        distances = torch.linalg.norm(trajectory[1:] - trajectory[:-1], dim=-1)
        trajectory_length = torch.sum(distances, dim=-1)
        # compare the fraction of the trajectory length with the ideal length
        if normalize:
            ideal_length = torch.linalg.norm(x1 - x0, dim=-1)
            trajectory_length = trajectory_length / ideal_length

        lengths.append(trajectory_length)

    lengths = torch.stack(lengths, dim=0)
    return lengths