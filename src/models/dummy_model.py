"""Dummy Model."""

import torch.nn as nn

from models.base_model import BaseModel
from utilities.data import AMASSBatch
from utilities.losses import mse


class DummyModel(BaseModel):
    """
    Dummy model.

    This is a dummy model. It provides basic implementations to demonstrate how more advanced
    models can be built.
    """

    def __init__(self, config):
        self.n_history = 10
        super(DummyModel, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the dummy model."""
        # In this model we simply feed the last time steps of the seed to a dense layer and
        # predict the targets directly.
        self.dense = nn.Linear(
            in_features=self.n_history * self.pose_size,
            out_features=self.config.target_seq_len * self.pose_size,
        )

    def forward(self, batch: AMASSBatch):
        """
        Forward pass.

        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {"seed": batch.poses[:, : self.config.seed_seq_len], "predictions": None}
        batch_size = batch.batch_size
        model_in = batch.poses[
            :, self.config.seed_seq_len - self.n_history : self.config.seed_seq_len
        ]
        pred = self.dense(model_in.reshape(batch_size, -1))
        model_out["predictions"] = pred.reshape(batch_size, self.config.target_seq_len, -1)
        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        Backward pass.

        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out["predictions"]
        targets = batch.poses[:, self.config.seed_seq_len :]

        total_loss = mse(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will
        # automatically be logged.
        loss_vals = {"total_loss": total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also
            # be called when evaluating the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
