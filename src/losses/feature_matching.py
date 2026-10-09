import torch
from transformers import HubertModel, AutoModel
from torch import Tensor
from src.losses.baseloss import BaseLossFunction
from src import ModelOutput, LossOutput, Batch
import torch.nn as nn
import torch._dynamo as dynamo
from src.lightning_modules.fake_module import FakeModule


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

        # disable gradient computation for the feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

        self.n_conv_layers = len(self.model.feature_extractor.conv_layers)
        self.n_transformer_layers = len(self.model.encoder.layers)

        assert (
            self.last_n_conv_layers <= self.n_conv_layers
        ), f"last_n_conv_layers must be less than or equal to {self.n_conv_layers}"
        assert (
            self.first_n_transformer_layers <= self.n_transformer_layers
        ), f"first_n_transformer_layers must be less than or equal to {self.n_transformer_layers}"

        print("Initialized HubertFeatureExtractor with the following configuration:")
        print(f"  - Last {self.last_n_conv_layers} / {self.n_conv_layers} convolutional layers")
        print(f"  - First {self.first_n_transformer_layers} / {self.n_transformer_layers} transformer layers")

    @dynamo.disable
    def forward(self, audio: Tensor) -> list[Tensor]:
        feature_list = []

        assert audio.dim() == 3 and audio.shape[1] == 1, "Audio tensor must have shape (batch_size, 1, sequence_length)"

        # Do not normalize audio in order to keep amplitude information
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


class FeatureMatchingLoss(BaseLossFunction):
    def __init__(
        self,
        feature_extractor: HubertFeatureExtractor,
        loss_fn: BaseLossFunction,
        normalized_layer_weights: bool = False,
        to_db: bool = False,
    ):
        super().__init__()
        feature_extractor.requires_grad_(False)
        feature_extractor.eval()
        self.feature_extractor = FakeModule(feature_extractor)
        self.loss_fn = loss_fn
        self.normalized_layer_weights = normalized_layer_weights
        self.to_db = to_db

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        self.feature_extractor.eval()

        target = batch["target"]
        output = model_output["output"]
        batch_size = target.shape[0]

        combined = torch.cat([target, output], dim=0)
        combined_features = self.feature_extractor(combined)

        real_features = [feat[:batch_size] for feat in combined_features]
        generated_features = [feat[batch_size:] for feat in combined_features]

        if self.normalized_layer_weights:
            # Compute element counts and normalize layer weights by element count
            element_counts = [real_feat.numel() for real_feat in real_features]
            total_elements = sum(element_counts)
            layer_weights = [count / total_elements for count in element_counts]
        else:
            layer_weights = [1 / len(real_features)] * len(real_features)

        loss: LossOutput = {}
        for weight, real_feat, gen_feat in zip(layer_weights, real_features, generated_features):
            layer_loss: LossOutput = self.loss_fn({"output": gen_feat}, {"target": real_feat})
            for key, value in layer_loss.items():
                loss[key] = loss.get(key, 0.0) + weight * value

        if self.to_db:
            loss["loss"] = 10 * torch.log10(loss["loss"] + 1e-8)

        return loss
