from src.losses import MSELoss
from src import ModelOutput, Batch
import torch


def test_mse_loss():
    loss_fn = MSELoss()
    batch = Batch(input=torch.randn(16, 10), target=torch.randn(16, 1))
    model_output = ModelOutput(output=torch.randn(16, 1))
    loss = loss_fn(model_output, batch)
    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"


def test_drifting_loss():
    from src.losses import DriftingLoss

    loss_fn = DriftingLoss(temperature=1.0)
    batch = Batch(input=torch.randn(4, 1, 16000), target=torch.randn(4, 1, 16000))
    model_output = ModelOutput(output=torch.randn(4, 1, 16000))
    model_output["output"].requires_grad = True

    loss = loss_fn(model_output, batch)

    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"

    # Check that gradient propagation works
    loss["loss"].backward()
    assert model_output["output"].grad is not None, "Gradients should propagate to model output"
    assert not torch.all(model_output["output"].grad == 0), "Gradients should be non-zero"


def test_drifting_loss_dynamic():
    from src.losses import DriftingLoss

    loss_fn = DriftingLoss(temperature=None)
    batch = Batch(input=torch.randn(4, 1, 16000), target=torch.randn(4, 1, 16000))
    model_output = ModelOutput(output=torch.randn(4, 1, 16000))
    model_output["output"].requires_grad = True

    loss = loss_fn(model_output, batch)

    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"

    # Check that gradient propagation works
    loss["loss"].backward()
    assert model_output["output"].grad is not None, "Gradients should propagate to model output"
    assert not torch.all(model_output["output"].grad == 0), "Gradients should be non-zero"


def test_hubert_feature_extractor_equality():
    from src.losses.feature_matching import HubertFeatureExtractor

    last_n_conv = 2
    first_n_trans = 3

    extractor = HubertFeatureExtractor(
        last_n_conv_layers=last_n_conv,
        first_n_transformer_layers=first_n_trans,
    )
    extractor.eval()

    audio = torch.randn(2, 1, 16000)

    # 1. Custom HubertFeatureExtractor output
    with torch.no_grad():
        custom_features = extractor(audio)

    assert (
        len(custom_features) == last_n_conv + first_n_trans
    ), f"Expected {last_n_conv + first_n_trans} feature maps, got {len(custom_features)}"

    # 2. Reference HubertModel forward pass on standardized input
    norm_audio = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(
        audio.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
    )
    raw_audio = norm_audio.squeeze(1)

    with torch.no_grad():
        model_output = extractor.model(raw_audio, output_hidden_states=True)

    # 3. Compare transformer outputs (the last first_n_trans feature maps)
    custom_transformer_feats = custom_features[last_n_conv:]
    ref_transformer_feats = model_output.hidden_states[1 : first_n_trans + 1]

    for layer_idx, (custom_feat, ref_feat) in enumerate(zip(custom_transformer_feats, ref_transformer_feats)):
        assert torch.allclose(
            custom_feat, ref_feat, atol=1e-5
        ), f"Feature mismatch at transformer layer {layer_idx + 1}"


def test_feature_matching_loss_with_fake_module():
    from src.losses.feature_matching import HubertFeatureExtractor, FeatureMatchingLoss
    from src.losses.baseloss import BaseLossFunction

    class DummyLoss(BaseLossFunction):
        def forward(self, model_output, batch):
            return {"l1": torch.abs(model_output["output"] - batch["target"]).mean()}

    extractor = HubertFeatureExtractor(last_n_conv_layers=1, first_n_transformer_layers=2)
    loss_fn = FeatureMatchingLoss(extractor, DummyLoss())

    batch = Batch(input=torch.randn(2, 1, 16000), target=torch.randn(2, 1, 16000))
    model_output = ModelOutput(output=torch.randn(2, 1, 16000))
    model_output["output"].requires_grad = True

    loss = loss_fn(model_output, batch)
    assert "l1" in loss
    assert loss["l1"].item() >= 0

    # Ensure fake module hides parameters from parameters() and state_dict()
    assert len(list(loss_fn.parameters())) == 0
    assert len(loss_fn.state_dict()) == 0

    # Ensure gradients flow back to model_output
    loss["l1"].backward()
    assert model_output["output"].grad is not None
    assert not torch.all(model_output["output"].grad == 0)


def test_power_loss():
    from src.losses import PowerLoss

    loss_fn = PowerLoss()

    # 1. Single channel (B, 1, T)
    target = torch.randn(4, 1, 16000)
    output = torch.randn(4, 1, 16000, requires_grad=True)

    batch = Batch(input=torch.randn(4, 1, 16000), target=target)
    model_output = ModelOutput(output=output)

    loss = loss_fn(model_output, batch)
    assert "loss" in loss
    assert loss["loss"].item() >= 0

    # Ensure identical signals yield 0 power loss
    ident_output = ModelOutput(output=target.clone())
    zero_loss = loss_fn(ident_output, batch)
    assert torch.isclose(zero_loss["loss"], torch.tensor(0.0), atol=1e-6)

    # Check gradient flow
    loss["loss"].backward()
    assert output.grad is not None
    assert not torch.all(output.grad == 0)

    # 2. Multi-channel audio shape (B, C, T) = (4, 2, 16000)
    loss_fn_bct = PowerLoss(dim=-1)
    target_bct = torch.randn(4, 2, 16000)
    output_bct = torch.randn(4, 2, 16000, requires_grad=True)
    batch_bct = Batch(input=torch.randn(4, 2, 16000), target=target_bct)
    model_out_bct = ModelOutput(output=output_bct)

    loss_bct = loss_fn_bct(model_out_bct, batch_bct)
    assert loss_bct["loss"].item() >= 0
    loss_bct["loss"].backward()
    assert output_bct.grad.shape == (4, 2, 16000)


def test_mel_spectrogram_feature_extractor():
    from src.losses import MelSpectrogramFeatureExtractor, FeatureMatchingLoss, L1Loss

    extractor = MelSpectrogramFeatureExtractor(
        sample_rate=16000,
        n_mels=80,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(128, 256, 512),
        win_lengths=(512, 1024, 2048),
    )

    audio = torch.randn(2, 1, 16000)
    features = extractor(audio)

    assert len(features) == 3, f"Expected 3 feature maps, got {len(features)}"
    assert features[0].dim() == 3 and features[0].shape[1] == 80

    # Test integration with FeatureMatchingLoss
    loss_fn = FeatureMatchingLoss(extractor, L1Loss())
    target = torch.randn(2, 1, 16000)
    output = torch.randn(2, 1, 16000, requires_grad=True)

    batch = Batch(input=audio, target=target)
    model_out = ModelOutput(output=output)

    loss_output = loss_fn(model_out, batch)
    assert "loss" in loss_output
    assert loss_output["loss"].item() >= 0

    # Test gradient flow back to output waveform
    loss_output["loss"].backward()
    assert output.grad is not None
    assert not torch.all(output.grad == 0)
