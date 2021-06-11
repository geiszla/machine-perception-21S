from models import GCN4attn
import numpy as np

import numpy as np
import torch
import torch.nn as nn

from models.base_model import BaseModel
from utilities.data import AMASSBatch
from utilities.losses import l1_loss
from utilities.utils import get_dct_matrix

class AttModel(BaseModel):

    def __init__(self,config):
        super(AttModel, self).__init__(config)

    def create_model(self, in_features=135, kernel_size=10, d_model=512, num_stage=12, dct_n=34):
        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN4attn.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.5,
                           num_stage=num_stage,
                           node_n=in_features,
                           config=self.config)

    def forward(self, batch: AMASSBatch, output_n=24, input_n=120):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        model_out = {"seed": batch.poses[:, : self.config.seed_seq_len], "predictions": None}
        src = batch.poses
        # print("src shape :" + str(src.shape))
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        # print("src_tmp shape :" + str(src_tmp.shape))
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        # print("src_key_tmp :"+str(src_key_tmp.shape))
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()
        # print("src_query_tmp :"+str(src_query_tmp.shape))

        dct_m, idct_m = get_dct_matrix(self.kernel_size + output_n)
        if torch.cuda.is_available():
            dct_m = torch.from_numpy(dct_m).float().cuda()
            idct_m = torch.from_numpy(idct_m).float().cuda()
        else:
            dct_m = torch.from_numpy(dct_m).float()
            idct_m = torch.from_numpy(idct_m).float()
        
        vn = input_n - self.kernel_size - output_n + 1
        # print(vn)
        vl = self.kernel_size + output_n
        # print(vl)
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        # print('src_value_tmp :' + str(src_value_tmp.shape))
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]
        # print('src_value_tmp :' + str(src_value_tmp.shape))

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        # print('key_tmp:'+str(key_tmp.shape))
        query_tmp = self.convQ(src_query_tmp / 1000.0)
        # print('query_tmp:'+str(query_tmp.shape))
        score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
        att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
        dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
            [bs, -1, dct_n])

        input_gcn = src_tmp[:, idx]
        dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                dct_out_tmp[:, :, :dct_n].transpose(1, 2))
        outputs.append(out_gcn)

        outputs = torch.cat(outputs, dim=2)
        # print(outputs.shape)
        model_out["predictions"] = outputs[:,-self.config.target_seq_len:]

        return model_out

    def backward(self, batch: AMASSBatch, model_out):

        predictions = model_out["predictions"]
        targets = batch.poses[:, self.config.seed_seq_len :]

        total_loss = l1_loss(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will
        # automatically be logged.
        loss_vals = {"total_loss": total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also
            # be called when evaluating the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
