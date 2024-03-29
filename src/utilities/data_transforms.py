"""
Data transformations to be applied to samples or batches before feeding the data to the model.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch

from utilities.data import AMASSSample


class ToTensor(object):
    """Convert numpy arrays inside samples to PyTorch tensors."""

    def __call__(self, sample: AMASSSample):
        """Convert numpy arrays inside samples to PyTorch tensors."""
        sample.to_tensor()
        return sample


class ExtractWindow(object):
    """
    Extract a window of a fixed size.

    If the sequence is shorter than the desired window size it will return the entire sequence
    without any padding.
    """

    def __init__(self, window_size, rng=None, mode="random"):
        """Initialize the extracted window."""
        assert mode in ["random", "beginning", "middle"]
        if mode == "random":
            assert rng is not None
        self.window_size = window_size
        self.rng = rng
        self.mode = mode
        self.padding_value = 0.0

    def __call__(self, sample: AMASSSample):
        """Extract the window of fixed size."""
        if sample.n_frames > self.window_size:
            if self.mode == "beginning":
                sf, ef = 0, self.window_size
            elif self.mode == "middle":
                mid = sample.n_frames // 2
                sf = mid - self.window_size // 2
                ef = sf + self.window_size
            elif self.mode == "random":
                sf = self.rng.randint(0, sample.n_frames - self.window_size + 1)
                ef = sf + self.window_size
            else:
                raise ValueError("Mode '{}' for window extraction unknown.".format(self.mode))
            return sample.extract_window(sf, ef)
        else:
            return sample


class DataAugmentation(object):
    """Data augmentation class."""

    def __call__(self, sample):
        """
        Call function.

        sample: AMASSSample
        """
        if torch.rand(1).item() > 0.5:
            poses = sample.poses.flip(0)
        else:
            poses = sample.poses
        return AMASSSample(sample.seq_id, poses)
