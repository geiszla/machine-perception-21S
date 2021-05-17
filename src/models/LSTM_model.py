"""LSTM Model."""

import torch.nn as nn
import torch
from models.base_model import BaseModel
from utilities.data import AMASSBatch
from utilities.losses import mse

class DenseLayer(nn.Module):
    """A fully connected layer with ReLU and BatchNorm"""
    def __init__(self,in_features, out_features):
        super(DenseLayer, self).__init__()
        self.fcn = nn.Linear(in_features,out_features)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_features)
    def forward(self,x):
        x = self.fcn(x)
        x = self.relu(x)
        # x = self.bn(x)
        return x

class LSTMModel(BaseModel):
    """LSTM model."""

    def __init__(self, config):
        self.n_history = 120
        super(LSTMModel, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """"Create the model."""
        self.LSTM = nn.LSTM(input_size = 135,
                            hidden_size = 135,
                            batch_first = True,
                            dropout = 0)
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
            # print(model_in.shape)
            out_LSTM = self.LSTM(model_in)[0][:,-1,:]
            # print(out_LSTM.shape)
            pred.append(out_LSTM.squeeze())
            model_in = torch.roll(model_in,-1,1)
            model_in[:,-1] = out_LSTM.squeeze()
        
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