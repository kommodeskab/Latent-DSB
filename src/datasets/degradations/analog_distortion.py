from src.datasets.degradations import BaseDegradation
import torch
from torch import Tensor, tanh
import torchaudio.functional as F


class AnalogDistortion(BaseDegradation):
    """
    Helper class for analog distortions. Based on the idea of harmonic distortion based on polynomial (leading taylor terms) of vinyl mechanics.
    Since amplitude distortion is just the discrepancy of the input/output curve of sound [1], we will model it like that. Digital audio typically clamps sound
    in the -1,1 range, which we do as well. The distortion might break down close to this point, but if the original audio ever reached this range, the original
    audio would have clipped as well, so distortions here do not really make sense anyway.

    We base the distortions on the following considerations;
    First, any distortion might be written as a taylor series,
    y(t) = a1 x(t) + a2 x(t)^2 + a3 x(t)^3 + ...
    Secondly, even order harmonics leads to assymetric transfer functions and odd order harmonics leads to symmetric transfer functions [1].
    Thirdly, we assume the following distortions
        1. Tracking Error
            As the tonearm of a vinyl is of a physical size and tracking angle, one side of the groove is read before the other, leading to an assymetric transfer function [2], contained mainly in
            even-order harmonics. [1]
        2. Tracing Distortion
            The vinyl is cut with a sharp, V-shaped lathe, while the sound is played back with a round stylus.
            This leads to symmetric distortions, since it happens equally on the positive and negative swings of the lateral groove, odd-order harmonics [1] [2].       
            The leading term in the distortion can be shown to scale by y(t) ~ x(t) + C x'(t)^2*x''(t), where C is a constant [3].
        3. Inner groove distortion
            In the inner grooves, the physical size of the vinyl is so small, that the physical size of the needle can't read it. This leads to a muffling of higher frequencies. Depending on the mass of the needle and cantilever,
            actually vibrating fast enough for the high frequencies is hard. This is achieved by just a low pass filter [2]
        4. RIAA equalization is a specification for the recording and playback of phonograph records, established by the Recording Industry Association of America (RIAA). [4]
           The implementation is based on
        5. We pass two highpass and low pass frequencies in the final end, to emulate really old historical data. for old, should be the defaults, otherwise
           change to something like 8000 and 10.

    Fourth, we add a drive parameter to emulate the electrical gain in a real vinyl. We add soft clipping with a tanh

    The two parameters to tune are a2 and a3, with given standard values. Suggested ranges are a2 ~ [0.1,1] a3 ~ [0.1, 1]


    Refs:
    [1] Alex Case, "Sound FX".
    [2] Some discussion with LLMs
    [3] A Theory of Tracing Distortion in Sound Reproduction from Phonograph Records, W. D. Lewis, F. V. Hunt
    [4] https://en.wikipedia.org/wiki/RIAA_equalization
    """

    def __init__(
        self,
        a2 = 0.2,
        a3 = 0.2,
        drive_db = 3.0,
        low_pass_freq = 3500,
        high_pass_freq = 200,
        sample_rate=16000,
        prob: float = 1.0,
        deterministic: bool = False,
    ):
        super().__init__(prob=prob, deterministic=deterministic)

        self.a2 = a2
        self.a3 = a3
        self.drive_db = drive_db
        self.low_pass_freq = low_pass_freq
        self.high_pass_freq = high_pass_freq
        self.sample_rate = sample_rate

    def _apply_riaa_pre(self, x: Tensor) -> Tensor:
        """
        Approximated RIAA pre-emphasis safe for arbitrary sample rates (e.g., 16kHz).
        - Cuts bass below 500.5 Hz (shelving naturally flattens out around 50.05 Hz)
        - Boosts treble above 2122.0 Hz
        """
        # ~17 dB cut to approximate the slope from 500Hz down to 50Hz
        x = F.bass_biquad(x, self.sample_rate, gain=-17.0, central_freq=500.5)
        # ~13.7 dB boost for the high frequencies
        x = F.treble_biquad(x, self.sample_rate, gain=13.7, central_freq=2122.0)
        return x

    def _apply_riaa_de(self, x: Tensor) -> Tensor:
        """
        Approximated RIAA de-emphasis (Playback EQ).
        - Boosts bass below 500.5 Hz
        - Cuts treble above 2122.0 Hz
        """
        x = F.bass_biquad(x, self.sample_rate, gain=17.0, central_freq=500.5)
        x = F.treble_biquad(x, self.sample_rate, gain=-13.7, central_freq=2122.0)
        return x

    def fun(self, audio: Tensor) -> Tensor:

        drive_linear = 10 ** (self.drive_db / 20.0)
        x = audio * drive_linear

        x_pre = self._apply_riaa_pre(x)

        tracking_dist = self.a2 * (x_pre ** 2)
        sr_ratio = 16000.0 / self.sample_rate

        v = torch.diff(x_pre, dim=-1, prepend=x_pre[..., :1]) * sr_ratio
        a = torch.diff(v, dim=-1, prepend=v[..., :1]) * sr_ratio
        
        tracing_dist = (self.a3 * (v ** 2) * a)

        y = x_pre + tracking_dist + tracing_dist

        y = F.lowpass_biquad(y, self.sample_rate,cutoff_freq=self.low_pass_freq)
        y = F.highpass_biquad(y,self.sample_rate,cutoff_freq=self.high_pass_freq)
        y = self._apply_riaa_de(y)
        return tanh(y)


if __name__ == "__main__":
    x: Tensor = torch.rand((1, 48000))
    vinyl = AnalogDistortion()
    x_distorted = vinyl(x)
    print(x_distorted.shape)
