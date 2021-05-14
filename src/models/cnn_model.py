"""Dummy Model."""

import torch.nn as nn
import torch
from models.base_model import BaseModel
from utilities.data import AMASSBatch
from utilities.losses import mse


class CNNModel(BaseModel):
    """
    Dummy model.

    This is a dummy model. It provides basic implementations to demonstrate how more advanced
    models can be built.
    """

    def __init__(self, config):
        self.n_history = 10
        super(CNNModel, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the dummy model."""
        # In this model we simply feed the last time steps of the seed to a dense layer and
        # predict the targets directly.
        self.dense = nn.Linear(
            in_features=self.n_history * self.pose_size,
            out_features=self.config.target_seq_len * self.pose_size,
        )
        
        self.cnn1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5,padding = 2)
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=8, kernel_size=5,padding = 2)
        self.cnn3 = nn.Conv1d(in_channels=8, out_channels = 4, kernel_size=5,padding = 2)
        self.cnn4 = nn.Conv1d(in_channels=4, out_channels = 1, kernel_size=5,padding = 2)

    def forward(self, batch: AMASSBatch):
        """
        Forward pass.

        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {"seed": batch.poses[:, : self.config.seed_seq_len], "predictions": None}
        batch_size = batch.batch_size
        model_in = batch.poses[
            :, self.config.seed_seq_len - 64 : self.config.seed_seq_len
        ]
        pred = []
        for i in range(self.config.target_seq_len):
            out_cnn1 = self.cnn1(model_in)
            # print(out_cnn1.shape)
            out_cnn2 = self.cnn2(out_cnn1)
            # print(out_cnn2.shape)
            out_cnn3 = self.cnn3(out_cnn2)
            # print(out_cnn3.shape)
            out_cnn4 = self.cnn4(out_cnn3)
            # print(out_cnn4.shape)
            pred.append(out_cnn4.squeeze())

            model_in = torch.roll(model_in,-1,1)
            # print(model_in[:,-1].shape)
            model_in[:,-1] = out_cnn4.squeeze()
        
        pred = torch.cat(pred)
        # print(pred.shape)
        model_out["predictions"] = pred.reshape(batch_size, self.config.target_seq_len, -1)
        # print(model_out["predictions"].shape)
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
