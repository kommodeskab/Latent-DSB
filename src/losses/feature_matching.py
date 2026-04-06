import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, HubertModel, AutoModel
from torch import Tensor
from src.losses.baseloss import BaseLossFunction
from src import AudioBatch, ModelOutput, LossOutput
import torch.nn as nn
from typing import Union


class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: Wav2Vec2Model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # extract these feature layers for calculating the loss
        self.layers_to_use = [0, 4, 8, 12]

    def __call__(self, audio: Tensor) -> list[Tensor]:
        # Normalize audio to have zero mean and unit variance
        audio = audio.squeeze(1)
        audio = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(audio.var(dim=-1, keepdim=True) + 1e-5)

        features = self.model.forward(audio, output_hidden_states=True).hidden_states
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
        self.n_transformer_layers = len(self.model.encoder.layers) + 1  # +1 for the feature projection layer
        assert (
            self.last_n_conv_layers <= self.n_conv_layers
        ), f"last_n_conv_layers must be less than or equal to {self.n_conv_layers}"
        assert (
            self.first_n_transformer_layers <= self.n_transformer_layers
        ), f"first_n_transformer_layers must be less than or equal to {self.n_transformer_layers}"

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


FEATURE_EXTRACTORS = Union[Wav2VecFeatureExtractor, HubertFeatureExtractor]


class FeatureMatchingLoss(BaseLossFunction):
    def __init__(self, feature_extractor: FEATURE_EXTRACTORS):
        super().__init__()
        self.feature_extractor = feature_extractor
        # make sure that the feature extractor is in eval mode and does not require gradients
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)

    def forward(self, model_output: ModelOutput, batch: AudioBatch) -> LossOutput:
        with torch.no_grad():
            real_features = self.feature_extractor(batch["target"])
        generated_features = self.feature_extractor(model_output["output"])

        loss = 0.0
        for real_feat, gen_feat in zip(real_features, generated_features):
            loss += F.l1_loss(gen_feat, real_feat)

        loss = loss / len(real_features)
        return LossOutput(loss=loss)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)

        keys_to_remove = [k for k in state_dict.keys() if k.startswith(f"{prefix}feature_extractor.")]
        for k in keys_to_remove:
            del state_dict[k]

        return state_dict

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        extractor_state = self.feature_extractor.state_dict(prefix=f"{prefix}feature_extractor.")
        state_dict.update(extractor_state)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
