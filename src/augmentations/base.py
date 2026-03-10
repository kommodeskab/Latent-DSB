from src import AudioSample


class BaseAugmentation:
    def __call__(self, sample: AudioSample) -> AudioSample:
        raise NotImplementedError("Subclasses must implement this method.")
