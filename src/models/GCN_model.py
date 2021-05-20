"""taken from https://github.com/wei-mao-2019/LearnTrajDep/blob/15c2bc5aafdbfae4e92f353ae19f00f7b12aea8c/utils/model.py."""

import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math

from models.base_model import BaseModel
from utilities.data import AMASSBatch
from utilities.losses import mse, l1_loss

def get_dct_matrix(N):
    """Outputs n*n matrix of DCT (Discrete Cosinus Transform) coefficients"""
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # print(input.shape)
        # print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)

        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModel(BaseModel):

    def __init__(self,config):
        super(GCNModel, self).__init__(config)

    def create_model(self):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        :param dct_n: number of kept DCT coefficients
        """
        input_feature = self.config.seed_seq_len + self.config.target_seq_len
        hidden_feature = 256
        p_dropout = 0.5
        node_n = 135
        self.num_stage = 12
        self.dct_n = 100

        self.idx_input = np.concatenate((np.arange(120),np.repeat(119,24)))
        dct_matrices = get_dct_matrix(input_feature)

        if torch.cuda.is_available():
            self.dct_matrix = torch.from_numpy(dct_matrices[0]).type(torch.float32).cuda()
            self.idct_matrix = torch.from_numpy(dct_matrices[1]).type(torch.float32).cuda()
        else:
            self.dct_matrix = torch.from_numpy(dct_matrices[0]).type(torch.float32)
            self.idct_matrix = torch.from_numpy(dct_matrices[1]).type(torch.float32)

        self.gc1 = GraphConvolution(self.dct_n, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(self.num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, self.dct_n, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, batch: AMASSBatch):
        model_out = {"seed": batch.poses[:, : self.config.seed_seq_len], "predictions": None}
        model_in = batch.poses[:,self.idx_input]

        #Getting DCT coefficients of input
        x = torch.matmul(self.dct_matrix[:self.dct_n,:],model_in).transpose(2,1)
        # print(x.shape)
        y = self.gc1(x)
        b, n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x
        y = y.transpose(2,1)
        y = torch.matmul(self.idct_matrix[:,:self.dct_n],y)
        model_out["predictions"] = y[:,-self.config.target_seq_len:]
        # print("pred shape")
        # print(model_out["predictions"].shape)
        model_out["full_pred"] = y
        return model_out
    
    def backward(self, batch: AMASSBatch, model_out):

        predictions = model_out["predictions"]
        targets = batch.poses[:,-self.config.target_seq_len:]

        total_loss = l1_loss(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will
        # automatically be logged.
        loss_vals = {"total_loss": total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also
            # be called when evaluating the model on the validation set.
            total_loss.backward()

        return loss_vals, targets[:,self.config.seed_seq_len:]