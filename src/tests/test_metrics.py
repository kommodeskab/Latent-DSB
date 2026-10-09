from unittest.mock import MagicMock
import numpy as np
import torch
from src import AudioBatch, StepOutput
from src.callbacks.metrics import LogAudioMetric


def test_log_audio_metric():
    metric = LogAudioMetric(key="waveform", where="batch", max_samples=2)

    mock_logger = MagicMock()
    mock_module = MagicMock()
    mock_module.logger = mock_logger
    mock_module.global_step = 10

    # Create unnormalized batch audio with amplitude exceeding 1 (peak = 5.0)
    waveform = torch.tensor(
        [
            [[0.0, 5.0, -2.5, 1.0]],
            [[0.0, -0.5, 0.25, 0.1]],
        ]
    )
    batch = AudioBatch(waveform=waveform, sample_rate=[16000, 16000])
    outputs = StepOutput()

    metric.add(
        pl_module=mock_module,
        outputs=outputs,
        batch=batch,
        batch_idx=0,
    )

    assert metric._is_full()
    # Ensure metric.samples contains the raw, unnormalized waveforms
    assert np.allclose(metric.samples[0], np.array([0.0, 5.0, -2.5, 1.0]))
    assert np.allclose(metric.samples[1], np.array([0.0, -0.5, 0.25, 0.1]))

    metric.compute()

    # Verify log_audio call
    assert mock_logger.log_audio.called
    audio_kwargs = mock_logger.log_audio.call_args.kwargs
    assert audio_kwargs["key"] == "log_audio/waveform"
    assert audio_kwargs["step"] == 10
    logged_audios = audio_kwargs["audios"]
    assert len(logged_audios) == 2

    # Check peak normalization: max absolute amplitude must be 1.0
    assert np.isclose(np.max(np.abs(logged_audios[0])), 1.0)
    assert np.allclose(logged_audios[0], np.array([0.0, 1.0, -0.5, 0.2]))
    assert np.isclose(np.max(np.abs(logged_audios[1])), 1.0)
    assert np.allclose(logged_audios[1], np.array([0.0, -1.0, 0.5, 0.2]))

    # Verify log_image call for the waveform plot
    assert mock_logger.log_image.called
    image_kwargs = mock_logger.log_image.call_args.kwargs
    assert image_kwargs["key"] == "log_waveform/waveform"
    assert image_kwargs["step"] == 10
    assert len(image_kwargs["images"]) == 1

    # Check reset
    metric.reset()
    assert len(metric.samples) == 0
    assert len(metric.sample_rates) == 0
