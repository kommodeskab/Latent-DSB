from src.augmentations.base import BaseAugmentation
from src import AudioSample
import torch


class CutAndFitAugmentation(BaseAugmentation):
    """
    An augmentation for cutting and fitting an audio sample to a specific length.
    If the input sample is too short, it will be padded with zeros on either side randomly.
    If the input sample is too long, it will be randomly cut to the target length.
    If deterministic is true, the padding and cutting will be done in a deterministic way (always pad/cut from the same side).
    """

    def __init__(
        self,
        target_length: int,
        deterministic: bool = False,
    ):
        self.target_length = target_length
        self.deterministic = deterministic

    def __call__(self, sample: AudioSample) -> AudioSample:
        input = sample["waveform"]
        length = input.shape[-1]

        if length == self.target_length:
            return sample

        if length < self.target_length:
            padding = self.target_length - length

            if self.deterministic:
                # always pad on the right side
                padded_input = torch.nn.functional.pad(input, (0, padding))

            else:
                # randomly pad on either side
                left_padding = torch.randint(0, padding + 1, (1,)).item()
                right_padding = padding - left_padding
                padded_input = torch.nn.functional.pad(input, (left_padding, right_padding))

            return AudioSample(waveform=padded_input, sample_rate=sample["sample_rate"])

        else:
            # length > target_length, need to cut
            if self.deterministic:
                # always cut from the right side
                cut_input = input[..., : self.target_length]

            else:
                # randomly cut from either side
                start = torch.randint(0, length - self.target_length + 1, (1,)).item()
                cut_input = input[..., start : start + self.target_length]

            return AudioSample(waveform=cut_input, sample_rate=sample["sample_rate"])


if __name__ == "__main__":
    # Example usage
    sample = AudioSample(
        waveform=torch.randn(1, 16000),  # 1 second of audio at 16kHz
        sample_rate=16000,
    )

    augmentation = CutAndFitAugmentation(target_length=8000, deterministic=False)
    augmented_sample = augmentation(sample)
    print(augmented_sample["waveform"].shape)  # should be (1, 8000)
