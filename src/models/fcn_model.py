"""FCN Model."""

import torch
import torch.nn as nn

from models.base_model import BaseModel
from utilities.data import AMASSBatch
from utilities.losses import mse


class DenseLayer(nn.Module):
    """A fully connected layer with ReLU and BatchNorm."""

    def __init__(self, in_features, out_features):
        super(DenseLayer, self).__init__()
        self.fcn = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """Forward function."""
        x = self.fcn(x)
        x = self.relu(x)
        # x = self.bn(x)
        return x


class FCNModel(BaseModel):
    """FCN model."""

    def __init__(self, config):
        self.n_history = 120
        super(FCNModel, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the model."""
        self.dense1 = DenseLayer(
            in_features=self.n_history * self.pose_size,
            out_features=256,
        )
        self.dense2 = DenseLayer(
            in_features=256,
            out_features=128,
        )
        self.dense3 = DenseLayer(
            in_features=128,
            out_features=64,
        )
        self.dense4 = DenseLayer(
            in_features=64,
            out_features=self.pose_size,
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
        pred = []
        for i in range(self.config.target_seq_len):
            out_dense1 = self.dense1(model_in.reshape(batch_size, -1))
            # print(out_cnn1.shape)
            out_dense2 = self.dense2(out_dense1)
            # print(out_cnn2.shape)
            out_dense3 = self.dense3(out_dense2)
            # print(out_cnn3.shape)
            out_dense4 = self.dense4(out_dense3)
            # print(out_cnn4.shape)
            pred.append(out_dense4.squeeze())

            model_in = torch.roll(model_in, -1, 1)
            # print(model_in[:,-1].shape)
            model_in[:, -1] = out_dense4.squeeze()

        pred_tensor = torch.cat(pred)
        # print(pred.shape)
        model_out["predictions"] = pred_tensor.reshape(batch_size, self.config.target_seq_len, -1)
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
