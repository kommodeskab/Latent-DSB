import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from torch import Tensor
from torchaudio.functional import resample
from src.losses.baseloss import BaseLossFunction
from src import AudioBatch, ModelOutput, LossOutput


class Wav2VecFeatureExtractor:
    def __init__(self):
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        # extract these feature layers for calculating the loss
        self.layers_to_use = [0, 4, 8, 12]
        self.target_sr = 16000
        
    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self
        
    def __call__(self, audio: Tensor, sample_rate: int) -> list[Tensor]:
        if sample_rate != self.target_sr:
            audio = resample(audio, orig_freq=sample_rate, new_freq=self.target_sr)
            
        # Normalize audio to have zero mean and unit variance
        audio = audio.squeeze(1)
        audio = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(audio.var(dim=-1, keepdim=True) + 1e-5)
            
        features = self.model.forward(audio, output_hidden_states=True).hidden_states
        return [features[i] for i in self.layers_to_use]


class FeatureMatchingLoss(BaseLossFunction):
    def __init__(self, sample_rate: int, compile: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.compile = compile
        self.feature_extractor = Wav2VecFeatureExtractor()
        
        if compile:
            self.feature_extractor.model = torch.compile(self.feature_extractor.model)
            
    def _apply(self, fn):
        # Called by nn.Module.to/cuda/cpu/half recursion from parent modules
        super()._apply(fn)

        probe = fn(torch.empty(0))
        self.feature_extractor.to(probe.device)
        return self
    
    def forward(self,  model_output: ModelOutput, batch: AudioBatch) -> LossOutput:        
        with torch.no_grad(): # make sure we don't compute gradients for the real features
            real_features = self.feature_extractor(batch["target"], self.sample_rate)
            
        generated_features = self.feature_extractor(model_output["output"], self.sample_rate)
        
        loss = 0.0
        for real_feat, gen_feat in zip(real_features, generated_features):
            loss += F.mse_loss(gen_feat, real_feat)
        
        loss = loss / len(real_features)
        return LossOutput(loss=loss)