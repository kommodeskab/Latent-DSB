import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, HubertModel, AutoModel
from torch import Tensor
from src.losses.baseloss import BaseLossFunction
from src import AudioBatch, ModelOutput, LossOutput


class BaseFeatureExtractor:
    model: Wav2Vec2Model | HubertModel

    def __call__(self, audio: Tensor) -> list[Tensor]: ...
    def to(self, device: torch.device): ...


class Wav2VecFeatureExtractor:
    def __init__(self):
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # extract these feature layers for calculating the loss
        self.layers_to_use = [0, 4, 8, 12]

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def __call__(self, audio: Tensor) -> list[Tensor]:
        # Normalize audio to have zero mean and unit variance
        audio = audio.squeeze(1)
        audio = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(audio.var(dim=-1, keepdim=True) + 1e-5)

        features = self.model.forward(audio, output_hidden_states=True).hidden_states
        return [features[i] for i in self.layers_to_use]


class HubertFeatureExtractor:
    def __init__(
        self,
        last_n_conv_layers: int,
        first_n_transformer_layers: int,
    ):
        self.last_n_conv_layers = last_n_conv_layers
        self.first_n_transformer_layers = first_n_transformer_layers

        self.model: HubertModel = AutoModel.from_pretrained("facebook/hubert-base-ls960")
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.n_conv_layers = len(self.model.feature_extractor.conv_layers)
        self.n_transformer_layers = len(self.model.encoder.layers) + 1  # +1 for the feature projection layer
        assert (
            self.last_n_conv_layers <= self.n_conv_layers
        ), f"last_n_conv_layers must be less than or equal to {self.n_conv_layers}"
        assert (
            self.first_n_transformer_layers <= self.n_transformer_layers
        ), f"first_n_transformer_layers must be less than or equal to {self.n_transformer_layers}"

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def __call__(self, audio: Tensor) -> list[Tensor]:
        feature_list = []

        assert audio.dim() == 3 and audio.shape[1] == 1, "Audio tensor must have shape (batch_size, 1, sequence_length)"
        # normalize audio using F.layer_norm
        extract_features = audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-5)

        n_conv_layers = len(self.model.feature_extractor.conv_layers)
        for n, conv_layer in enumerate(self.model.feature_extractor.conv_layers):
            extract_features = conv_layer(extract_features)
            if n >= n_conv_layers - self.last_n_conv_layers:
                feature_list.append(extract_features)

        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.model.feature_projection(extract_features)
        hidden_states = self.model._mask_hidden_states(hidden_states, mask_time_indices=None)

        encoder_outputs = self.model.encoder.forward(
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = encoder_outputs.hidden_states[: self.first_n_transformer_layers]
        feature_list.extend(hidden_states)

        return feature_list


class FeatureMatchingLoss(BaseLossFunction):
    def __init__(self, feature_extractor: BaseFeatureExtractor, compile: bool = False):
        super().__init__()
        self.feature_extractor = feature_extractor

        if compile:
            self.feature_extractor.model = torch.compile(self.feature_extractor.model)

    def _apply(self, fn):
        # Called by nn.Module.to/cuda/cpu/half recursion from parent modules
        super()._apply(fn)

        probe = fn(torch.empty(0))
        self.feature_extractor.to(probe.device)
        return self

    def forward(self, model_output: ModelOutput, batch: AudioBatch) -> LossOutput:
        with torch.no_grad():
            real_features = self.feature_extractor(batch["target"])
        generated_features = self.feature_extractor(model_output["output"])

        loss = 0.0
        for real_feat, gen_feat in zip(real_features, generated_features):
            loss += F.l1_loss(gen_feat, real_feat)

        loss = loss / len(real_features)
        return LossOutput(loss=loss)
