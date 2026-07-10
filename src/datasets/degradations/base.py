from torch import Tensor


class BaseDegradation:
    """
    Base class for audio degradations.

    Args:
        prob (float): Probability of applying degradation. Must be between 0 and 1.
        deterministic (bool): Whether to use deterministic behavior for reproducibility.
    """

    def fun(self, audio: Tensor) -> Tensor:
        raise NotImplementedError("Degradation function not implemented")

    def __call__(self, audio: Tensor) -> Tensor:
        return self.fun(audio)
