import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src import AudioSample
from src.augmentations import NormalizeAugmentation


def test_normalize_augmentation():
    # Create a random audio sample with non-zero mean and non-unit std
    torch.manual_seed(42)
    waveform = torch.randn(1, 16000) * 4.5 + 2.0
    sample = AudioSample(waveform=waveform, sample_rate=16000)

    aug = NormalizeAugmentation()
    augmented = aug(sample)

    assert augmented["sample_rate"] == 16000
    assert torch.isclose(augmented["waveform"].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(augmented["waveform"].std(), torch.tensor(1.0), atol=1e-5)


def test_normalize_augmentation_constant_signal():
    # Test with constant (silence) signal to ensure eps avoids division by zero and produces finite values
    waveform = torch.zeros(1, 16000)
    sample = AudioSample(waveform=waveform, sample_rate=16000)

    aug = NormalizeAugmentation()
    augmented = aug(sample)

    assert not torch.isnan(augmented["waveform"]).any()
    assert not torch.isinf(augmented["waveform"]).any()


def test_normalize_augmentation_hydra():
    config = OmegaConf.create({"_target_": "src.augmentations.NormalizeAugmentation"})
    aug = instantiate(config)
    assert isinstance(aug, NormalizeAugmentation)
