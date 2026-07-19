import torch
from transformers import Wav2Vec2Model, HubertModel, AutoModel
from torch import Tensor
from src.losses.baseloss import BaseLossFunction
from src import ModelOutput, LossOutput, Batch
import torch.nn as nn
import torch._dynamo as dynamo
from typing import Union

class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: Wav2Vec2Model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # extract these feature layers for calculating the loss
        self.layers_to_use = [0, 4, 8, 12]

    def forward(self, audio: Tensor) -> list[Tensor]:
        # Normalize audio to have zero mean and unit variance
        audio = audio.squeeze(1)
        audio = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(audio.var(dim=-1, keepdim=True) + 1e-5)

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

        # audio_var = audio.var(dim=-1, unbiased=False, keepdim=True)
        # extract_features = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(audio_var + 1e-5)
        extract_features = audio

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


FEATURE_EXTRACTORS = Union[Wav2VecFeatureExtractor, HubertFeatureExtractor]


class FeatureMatchingLoss(BaseLossFunction):
    def __init__(
        self, 
        feature_extractor: FEATURE_EXTRACTORS,
        loss_fn: BaseLossFunction
        ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn
        # make sure that the feature extractor is in eval mode and does not require gradients
        self.feature_extractor.requires_grad_(False)

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        self.feature_extractor.eval()
        real_features = self.feature_extractor(batch["target"])
        generated_features = self.feature_extractor(model_output["output"])
        
        for layer_idx, (real_feat, gen_feat) in enumerate(zip(real_features, generated_features)):
            layer_loss = self.loss_fn(
                {"output": gen_feat},
                {"target": real_feat}
            )
            
            if layer_idx == 0:
                # initialize the loss dictionary on the first iteration
                loss = layer_loss
                
            else:
                # for the other iterations, accumulate the loss values for each key
                for key, value in layer_loss.items():
                    loss[key] += value

        # average the loss values over the number of layers
        loss = {key: value / len(real_features) for key, value in loss.items()}
        
        return loss

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # When saving the state dict, DONT include the weights ofn the feature extractor
        # why? Because the feature extractor is a large but frozen model that we don't need to save.
        # therefore, we save space and decrease complexity

        state_dict = super().state_dict(destination, prefix, keep_vars)

        keys_to_remove = [k for k in state_dict.keys() if k.startswith(f"{prefix}feature_extractor.")]
        for k in keys_to_remove:
            del state_dict[k]

        return state_dict

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # When loading the state dict, we need to load the weights of the feature extractor from the state dict
        # they are not included in the state dict, see the state_dict method

        extractor_state = self.feature_extractor.state_dict(prefix=f"{prefix}feature_extractor.")
        state_dict.update(extractor_state)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
