from src.datasets.degradations import BaseDegradation
import torch
from torch import Tensor
import torchaudio.functional as F
from typing import Literal


class AnalogNoise(BaseDegradation):
    """
    Helper class for emulating clicking, crackling, scratching and tick like noise associated with analogue disc recording methods
    such as vinyl.

    Models noise of the form y_t = x_t + i_t * n_t, where i_t is an on/off switching of the noise and n_t is some distribution of noise. [1]
    If the noise term n_t is modelled as a harmonic oscillator, the clicking can be realised using a bandpass filter for the finite difference model. [2]

    Each click has a peak power with a randomized peak with a loudness between min_db and max_db (compared to a dbfs level of 1.0), meaning that
    both min_db and max_db MUST be negative.

    * Clicks are larger pieces of debris or small groove faults in a vinyl - modelled with a medium high click rate, e.g. 5, a central frequency
    of ~1500-3000Hz, and a medium high Q-factor, e.g. ~1-3

    * Crackle is smaller debris like dust or small defects - modelled with a very high click rate, e.g. ~50-200, a lower q-factor like ~0.1-0.5, and a
    higher central frequency e.g. ~4000-6000Hz

    * Scratching is large defects with a low rate, e.g. 0.1-2. A more "thuddy" or "thumpy" sound is realised with lower central frequency ~1000-2000Hz.
    Since they are large defects, a higher Q-factor is used, i.e. ~2

    * Ticking arises from small vinyl tip oscillations, having a higher frequency than dust. Air bubble in the material, or static discharge i.e.
    Choose here a much higher frequency, ~7000-7500. The Q-facotr is ~1-1.5. The click rate is ~1-5.

    * Thumping might arise from footsteps near the vinyl player, or large scale warping of the disc. Very low frequency, mid-low Q-factor, high amplitude oscillations.
    Low click rate.

    * If noise type is chosen as None, manual selection of parameters can be done.

    Refs:
    [1] Simon J. Godswill and Peter J. W. Rayner, "Digital Audio Restoration", chapter 5.
    [2] Julius O. Smith III, "Physical Audio Signal Processing", chapter on "Lumped Models", https://ccrma.stanford.edu/~jos/pasp/Lumped_Models.html
    [3] Some discussion with LLMs
    """

    def __init__(
        self,
        min_db=None,
        max_db=None,
        clicks_per_second=None,  # click rate
        central_frequency=None,  # How high pitched is the clicking/crackling - essentially the mass of the needle/springiness of the needle arm.
        Q_factor=None,  # For how long does the impulse last - for dust, use a low factor, for scratches, use a high factor.
        noise_type: Literal["Click", "Crackle", "Scratch", "Tick", "Thump"] = "Click",
        sample_rate=16000,  # needed for the poisson process. Don't expect a hard call, but maybe.
        prob: float = 1.0,
        deterministic: bool = False,
    ):
        super().__init__(prob=prob, deterministic=deterministic)

        _min_db = -30.0
        _max_db = -10.0
        _clicks_per_second = 5.0
        _central_frequency = 4000.0
        _Q_factor = 1.0

        match noise_type:
            case "Click":
                _min_db = -15.0
                _max_db = -5.0
                _clicks_per_second = 5
                _central_frequency = 2500.0
                _Q_factor = 2.5
            case "Crackle":
                _min_db = -40.0
                _max_db = -30.0
                _clicks_per_second = 150
                _central_frequency = 5000.0
                _Q_factor = 0.75
            case "Scratch":
                _min_db = -6.0
                _max_db = -2.0
                _clicks_per_second = 0.75
                _central_frequency = 1000.0
                _Q_factor = 3.0
            case "Tick":
                _min_db = -25.0
                _max_db = -15.0
                _clicks_per_second = 10.0
                _central_frequency = 7500.0
                _Q_factor = 1.75
            case "Thump":
                _min_db = -10.0
                _max_db = -2.0
                _clicks_per_second = 0.75
                _central_frequency = 80
                _Q_factor = 0.5
            case None:
                pass

        self.min_db = min_db if min_db is not None else _min_db
        self.max_db = max_db if max_db is not None else _max_db
        self.clicks_per_second = clicks_per_second if clicks_per_second is not None else _clicks_per_second
        self.central_frequency = central_frequency if central_frequency is not None else _central_frequency
        self.Q_factor = Q_factor if Q_factor is not None else _Q_factor
        self.sample_rate = sample_rate

    def _sample_impulses(self, num_clicks) -> Tensor:
        db_amps = torch.empty(num_clicks).uniform_(self.min_db, self.max_db)
        linear_amps = 10 ** (db_amps / 20.0)
        polarities = (
            torch.randint(0, 2, (num_clicks,)).float() * 2 - 1
        )  # defect can affect both positively and negatively
        return linear_amps * polarities

    def fun(self, audio: Tensor) -> Tensor:
        num_samples = audio.shape[-1]
        duration = num_samples / self.sample_rate
        # Find number of clicks from click rate and sample duration
        num_clicks = int(torch.poisson(torch.tensor(self.clicks_per_second * duration)).item())
        # Find click indices
        click_indices = torch.randperm(num_samples)[:num_clicks]
        # scatter impulses at random indices and at random amplitudes
        impulses = torch.zeros(num_samples)
        impulse_amps = self._sample_impulses(num_clicks)
        impulses.scatter_(0, click_indices, impulse_amps)
        # Shape the impulses according to a damped harmonic oscillator, using a biquad bandpass filter
        shaped_impulses = F.bandpass_biquad(impulses, self.sample_rate, self.central_frequency, self.Q_factor)
        output = audio + shaped_impulses
        return output.clamp(min=-1.0, max=1.0)


if __name__ == "__main__":
    x: Tensor = torch.rand((1, 48000))
    vinyl = AnalogNoise(min_db=-40, max_db=-10)
    x_clicked = vinyl(x)
    print(x_clicked.shape)
