"""
Base Model class, which all models will inherit from
"""
import torch.nn as nn

from utilities.data import AMASSBatch
from utilities.losses import mse

class BaseModel(nn.Module):
    """
    A base class for neural networks.

    A base class for neural networks that defines an interface and implements a few common
    functions.
    """

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size
        self.create_model()

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the model, called automatically by the initializer."""
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, batch: AMASSBatch):
        """Forward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, batch: AMASSBatch, model_out):
        """Backward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def model_name(self):
        """Print summary string of this model. Override this if desired."""
        return "{}-lr{}".format(self.__class__.__name__, self.config.lr)