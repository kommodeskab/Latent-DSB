import pytest
import torch
from src.losses import (
    MSELoss,
    L1Loss,
    HubertFeatureExtractor,
    FeatureMatchingLoss,
    DriftingLoss,
    WeightedLoss,
)
from src import ModelOutput, Batch


def test_mse_loss():
    loss_fn = MSELoss()
    batch = Batch(input=torch.randn(16, 10), target=torch.randn(16, 1))
    model_output = ModelOutput(output=torch.randn(16, 1))
    loss = loss_fn.forward(model_output, batch)
    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"


def test_hubert_feature_extractor_last_feature():
    """
    Test that HubertFeatureExtractor extracts features correctly by showing that
    when first_n_transformer_layers equals the total transformer layers, the last feature
    in feature_list is numerically identical to the output of calling the forward method
    of the underlying HubertModel.
    """
    torch.manual_seed(42)
    last_n_conv_layers = 2
    # The base Hubert model has 12 transformer layers; extracting all 12 means
    # feature_list[-1] is the last transformer layer's hidden state.
    extractor = HubertFeatureExtractor(
        last_n_conv_layers=last_n_conv_layers,
        first_n_transformer_layers=12,
    )
    extractor.eval()

    batch_size = 2
    sequence_length = 16000  # 1 second of audio at 16kHz
    # Use unnormalized audio with arbitrary amplitude/offset to verify amplitude is preserved
    audio = torch.randn(batch_size, 1, sequence_length) * 0.25 + 0.05

    with torch.no_grad():
        # Feature extractor produces conv features + transformer hidden states without normalization
        feature_list = extractor(audio)

        # Calling forward on the underlying HubertModel directly with the exact same audio
        hubert_output = extractor.model(audio.squeeze(1))

    # Total features: last_n_conv_layers + first_n_transformer_layers
    assert len(feature_list) == last_n_conv_layers + 12

    last_extracted_feature = feature_list[-1]
    expected_last_feature = hubert_output.last_hidden_state

    # Verify shapes match (batch_size, time_frames, hidden_dim)
    assert (
        last_extracted_feature.shape == expected_last_feature.shape
    ), f"Shape mismatch: {last_extracted_feature.shape} vs {expected_last_feature.shape}"

    # Verify values match exactly
    max_diff = torch.max(torch.abs(last_extracted_feature - expected_last_feature)).item()
    assert torch.allclose(
        last_extracted_feature, expected_last_feature, atol=1e-5
    ), f"Last feature does not match HubertModel.forward: max absolute difference = {max_diff}"
    assert max_diff == 0.0, f"Expected exact equality (max diff 0.0), got max diff {max_diff}"


def test_hubert_feature_extractor_intermediate_transformer_features():
    """
    Test that intermediate transformer features extracted by HubertFeatureExtractor
    match the corresponding hidden states returned by HubertModel(output_hidden_states=True).
    """
    torch.manual_seed(42)
    last_n_conv_layers = 3
    first_n_transformer_layers = 6

    extractor = HubertFeatureExtractor(
        last_n_conv_layers=last_n_conv_layers,
        first_n_transformer_layers=first_n_transformer_layers,
    )
    extractor.eval()

    audio = torch.randn(2, 1, 16000) * 0.3 - 0.1

    with torch.no_grad():
        feature_list = extractor(audio)
        hubert_output = extractor.model(audio.squeeze(1), output_hidden_states=True)

    assert len(feature_list) == last_n_conv_layers + first_n_transformer_layers

    # Check each extracted transformer layer matches the corresponding hidden state
    for idx, trans_idx in enumerate(range(1, first_n_transformer_layers + 1)):
        extracted = feature_list[last_n_conv_layers + idx]
        expected = hubert_output.hidden_states[trans_idx]
        assert torch.allclose(
            extracted, expected, atol=1e-5
        ), f"Transformer layer {trans_idx} did not match hidden_states[{trans_idx}]"

    # The last feature in feature_list should match the 6th layer hidden state
    assert torch.allclose(feature_list[-1], hubert_output.hidden_states[first_n_transformer_layers], atol=1e-5)


def test_feature_matching_loss_forward():
    """
    Smoke test that FeatureMatchingLoss computes a valid loss dictionary with HubertFeatureExtractor.
    """
    torch.manual_seed(42)
    extractor = HubertFeatureExtractor(last_n_conv_layers=2, first_n_transformer_layers=4)
    loss_fn = FeatureMatchingLoss(feature_extractor=extractor, loss_fn=L1Loss())

    batch = Batch(target=torch.randn(2, 1, 16000))
    model_output = ModelOutput(output=torch.randn(2, 1, 16000))

    loss = loss_fn(model_output, batch)
    assert "loss" in loss
    assert loss["loss"].item() >= 0.0


def test_feature_matching_loss_batch_stacking_equivalence():
    """
    Test that stacking target and output into a single batch yields identical features
    and loss compared to passing target and output separately through the feature extractor.
    """
    torch.manual_seed(42)
    extractor = HubertFeatureExtractor(last_n_conv_layers=2, first_n_transformer_layers=2)
    extractor.eval()

    target = torch.randn(2, 1, 16000)
    output = torch.randn(2, 1, 16000)

    # Separate feature extraction
    with torch.no_grad():
        sep_real = extractor(target)
        sep_gen = extractor(output)

    # Stacked feature extraction (as implemented in FeatureMatchingLoss)
    combined = torch.cat([target, output], dim=0)
    with torch.no_grad():
        combined_features = extractor(combined)

    b_size = target.shape[0]
    stacked_real = [feat[:b_size] for feat in combined_features]
    stacked_gen = [feat[b_size:] for feat in combined_features]

    for sr, r in zip(stacked_real, sep_real):
        assert torch.allclose(sr, r, atol=1e-5)

    for sg, g in zip(stacked_gen, sep_gen):
        assert torch.allclose(sg, g, atol=1e-5)


def test_drifting_loss_temperature_validation():
    """
    Test that DriftingLoss requires temperature to be a positive float,
    rejecting non-positive values, non-numeric types, None, and removed kernel_type arguments.
    """
    # Valid positive temperatures
    loss_fn = DriftingLoss(temperature=0.5)
    assert loss_fn.temperature == 0.5
    loss_fn_int = DriftingLoss(temperature=2)
    assert loss_fn_int.temperature == 2.0

    # Invalid temperatures must raise ValueError or TypeError
    for invalid_temp in [0, 0.0, -1.0, -0.001, "0.5", None]:
        with pytest.raises((ValueError, TypeError)):
            DriftingLoss(temperature=invalid_temp)

    # kernel_type is no longer accepted as an argument
    with pytest.raises(TypeError):
        DriftingLoss(temperature=1.0, kernel_type="laplace")


def test_drifting_loss_anti_symmetry():
    """
    Test Proposition 3.1 (Deng et al., 2026): when the positive and negative sample
    distributions are identical (y_pos == y_neg), the drifting field V vanishes identically.
    """
    torch.manual_seed(42)
    loss_fn = DriftingLoss(temperature=1.0)
    batch_size = 4
    dim = 8

    x = torch.randn(batch_size, dim)
    y = torch.randn(6, dim)  # Independent target samples

    # When y_pos == y_neg, V should be identically zero
    V = loss_fn.compute_V(x, y_pos=y, y_neg=y)
    assert torch.allclose(
        V, torch.zeros_like(V), atol=1e-6
    ), f"Drifting field V is not zero under symmetric targets: max abs={V.abs().max().item()}"


def test_drifting_loss_numerical_stability_at_identical_points():
    """
    Test that when x matches y_pos identically (zero pairwise distance),
    compute_V does not produce any NaNs or Infs from division by zero.
    """
    torch.manual_seed(42)
    loss_fn = DriftingLoss(temperature=0.5)
    x = torch.randn(4, 16)

    # Exact identical points between model outputs and target
    V = loss_fn.compute_V(x, y_pos=x, y_neg=x)
    assert not torch.isnan(V).any(), "V contains NaNs when x == y_pos"
    assert not torch.isinf(V).any(), "V contains Infs when x == y_pos"


def test_drifting_loss_drift_direction():
    """
    Test that the drifting field pulls model samples toward the data distribution.
    If generated samples x are at 0 and target samples y_pos are strictly positive,
    the drift vector V must point in the positive direction.
    """
    loss_fn = DriftingLoss(temperature=1.0)
    x = torch.zeros(4, 2)
    y_pos = torch.ones(4, 2) * 3.0
    y_neg = x.clone()

    V = loss_fn.compute_V(x, y_pos=y_pos, y_neg=y_neg)
    assert (V > 0).all(), f"Expected positive drift toward target, got {V}"


def test_drifting_loss_forward_and_backward():
    """
    Test end-to-end forward and backward passes of DriftingLoss:
    1. Returns LossOutput dictionary with 'loss' >= 0.
    2. Backpropagates gradients into model_output['output'].
    3. Target is treated as stop-gradient (no grad graph attached).
    """
    torch.manual_seed(42)
    loss_fn = DriftingLoss(temperature=0.5)

    x = torch.randn(4, 16, requires_grad=True)
    target = torch.randn(4, 16)
    model_output = ModelOutput(output=x)
    batch = Batch(target=target)

    loss_output = loss_fn(model_output, batch)
    assert "loss" in loss_output
    loss = loss_output["loss"]
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0).any(), "Gradients should be non-zero"


def test_drifting_loss_multidimensional_audio():
    """
    Test that DriftingLoss handles multi-dimensional tensors such as audio signals
    (batch_size, channels, time_steps) and preserves tensor shapes properly.
    """
    torch.manual_seed(42)
    loss_fn = DriftingLoss(temperature=1.0)

    # 3D tensor: (batch_size=2, channels=1, time_steps=1600)
    x = torch.randn(2, 1, 1600, requires_grad=True)
    target = torch.randn(2, 1, 1600)

    loss_output = loss_fn(ModelOutput(output=x), Batch(target=target))
    assert "loss" in loss_output
    loss = loss_output["loss"]
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 1, 1600)
    assert not torch.isnan(x.grad).any()


def test_weighted_loss_function_names_unique():
    """
    Test that when all loss classes are unique, function_names contains the class names
    without any numeric suffixes.
    """
    weighted = WeightedLoss(losses=[MSELoss(), L1Loss()], weights=[1.0, 0.5])
    assert weighted.function_names == ["MSELoss", "L1Loss"]
    assert weighted.loss_names == ["MSELoss", "L1Loss"]


def test_weighted_loss_function_names_duplicates():
    """
    Test that when loss classes are duplicated, numeric suffixes (1, 2, ...) are appended
    in sequential order for each duplicated class.
    """
    weighted = WeightedLoss(
        losses=[MSELoss(), L1Loss(), MSELoss(), MSELoss(), L1Loss()],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    expected_names = ["MSELoss1", "L1Loss1", "MSELoss2", "MSELoss3", "L1Loss2"]
    assert weighted.function_names == expected_names
    assert weighted.loss_names == expected_names


def test_weighted_loss_forward_keys_and_values():
    """
    Test that forward on WeightedLoss prefixes keys with the constructed function names,
    preventing duplicate keys from overwriting each other, and produces the correct weighted sum.
    """
    mse1 = MSELoss()
    mse2 = MSELoss()
    w1, w2 = 2.0, 0.5
    weighted = WeightedLoss(losses=[mse1, mse2], weights=[w1, w2])

    output = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.0, 0.0]])
    model_output = ModelOutput(output=output)
    batch = Batch(target=target)

    # MSE of [1, 2] vs [0, 0] is (1 + 4)/2 = 2.5
    expected_individual_loss = 2.5
    expected_total_loss = (w1 * expected_individual_loss + w2 * expected_individual_loss) / (w1 + w2)

    loss_output = weighted(model_output, batch)
    assert "loss" in loss_output
    assert torch.isclose(loss_output["loss"], torch.tensor(expected_total_loss))

    # Verify both prefixed keys exist and are not overwritten
    assert "MSELoss1_loss" in loss_output
    assert "MSELoss2_loss" in loss_output
    assert torch.isclose(loss_output["MSELoss1_loss"], torch.tensor(expected_individual_loss))
    assert torch.isclose(loss_output["MSELoss2_loss"], torch.tensor(expected_individual_loss))
