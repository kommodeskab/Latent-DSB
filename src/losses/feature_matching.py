import torch
from transformers import Wav2Vec2Model, HubertModel, AutoModel
from torch import Tensor
from src.losses.baseloss import BaseLossFunction
from src import ModelOutput, LossOutput, Batch
import torch.nn as nn
import torch._dynamo as dynamo
from typing import Union
from src.lightning_modules.fake_module import FakeModule
import torchaudio.transforms as T
from typing import Tuple


class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: Wav2Vec2Model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # extract these feature layers for calculating the loss
        self.layers_to_use = [0, 4, 8, 12]

    def forward(self, audio: Tensor) -> list[Tensor]:
        # Ensure 2D tensor (batch_size, sequence_length) for Wav2Vec2Model
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        # Normalize audio to have zero mean and unit variance
        audio = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(
            audio.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
        )

        features = self.model(audio, output_hidden_states=True).hidden_states
        return [features[i] for i in self.layers_to_use]


class HubertFeatureExtractor(nn.Module):
    def __init__(
        self,
        last_n_conv_layers: int,
        first_n_transformer_layers: int,
    ):
        super().__init__()
        self.last_n_conv_layers = last_n_conv_layers
        self.first_n_transformer_layers = first_n_transformer_layers

        self.model: HubertModel = AutoModel.from_pretrained("facebook/hubert-base-ls960")

        self.n_conv_layers = len(self.model.feature_extractor.conv_layers)
        self.n_transformer_layers = len(self.model.encoder.layers)

        assert (
            self.last_n_conv_layers <= self.n_conv_layers
        ), f"last_n_conv_layers must be less than or equal to {self.n_conv_layers}"
        assert (
            self.first_n_transformer_layers <= self.n_transformer_layers
        ), f"first_n_transformer_layers must be less than or equal to {self.n_transformer_layers}"

    @dynamo.disable
    def forward(self, audio: Tensor) -> list[Tensor]:
        feature_list = []

        assert audio.dim() == 3 and audio.shape[1] == 1, "Audio tensor must have shape (batch_size, 1, sequence_length)"

        # Normalize audio to zero-mean and unit-variance per utterance (standard for HuBERT / Wav2Vec2)
        extract_features = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(
            audio.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
        )

        n_conv_layers = len(self.model.feature_extractor.conv_layers)
        for n, conv_layer in enumerate(self.model.feature_extractor.conv_layers):
            extract_features = conv_layer(extract_features)
            if n >= n_conv_layers - self.last_n_conv_layers:
                # Transpose immediately so all features in the list share the (Batch, Time, Channels) layout
                feature_list.append(extract_features.transpose(1, 2))

        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.model.feature_projection(extract_features)
        hidden_states = self.model._mask_hidden_states(hidden_states, mask_time_indices=None)

        encoder_outputs = self.model.encoder(
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        transformer_outputs = encoder_outputs.hidden_states[1 : self.first_n_transformer_layers + 1]
        feature_list.extend(transformer_outputs)

        return feature_list


class MelSpectrogramFeatureExtractor(nn.Module):
    """
    Computes multi-resolution Mel Spectrograms for feature matching.

    Returns a list of log-mel spectrogram tensors computed at different STFT resolutions
    (e.g., varying n_fft, win_length, hop_length, and n_mels).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_lengths: Tuple[int, ...] = (512, 1024, 2048),
        log_scale: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert (
            len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        ), "fft_sizes, hop_sizes, and win_lengths must have equal length"

        self.sample_rate = sample_rate
        self.log_scale = log_scale
        self.eps = eps

        # Store transforms in nn.ModuleList so device placement works (.to, .cuda, etc.)
        self.mel_transforms = nn.ModuleList(
            [
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    win_length=win_len,
                    hop_length=hop_len,
                    n_mels=n_mels,
                    power=1.0,
                )
                for n_fft, hop_len, win_len in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, audio: Tensor) -> list[Tensor]:
        # Ensure 2D tensor (batch_size, sequence_length) for torchaudio
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        feature_list = []
        for transform in self.mel_transforms:
            mel = transform(audio)
            if self.log_scale:
                mel = torch.log(torch.clamp(mel, min=self.eps))
            feature_list.append(mel)

        return feature_list


FEATURE_EXTRACTORS = Union[Wav2VecFeatureExtractor, HubertFeatureExtractor, MelSpectrogramFeatureExtractor]


class FeatureMatchingLoss(BaseLossFunction):
    def __init__(self, feature_extractor: FEATURE_EXTRACTORS, loss_fn: BaseLossFunction):
        super().__init__()
        feature_extractor.requires_grad_(False)
        feature_extractor.eval()
        self.feature_extractor = FakeModule(feature_extractor)
        self.loss_fn = loss_fn

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        self.feature_extractor.eval()

        # Target feature extraction does not need autograd tracking
        with torch.no_grad():
            real_features = self.feature_extractor(batch["target"])

        generated_features = self.feature_extractor(model_output["output"])

        # Compute element counts and normalize layer weights by element count
        element_counts = [real_feat.numel() for real_feat in real_features]
        total_elements = sum(element_counts)
        layer_weights = [count / total_elements for count in element_counts]

        loss = {}
        for weight, real_feat, gen_feat in zip(layer_weights, real_features, generated_features):
            layer_loss = self.loss_fn({"output": gen_feat}, {"target": real_feat})
            for key, value in layer_loss.items():
                loss[key] = loss.get(key, 0.0) + weight * value

        return loss
