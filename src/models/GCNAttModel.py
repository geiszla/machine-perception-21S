from models import GCN4attn
import numpy as np

import numpy as np
import torch
import torch.nn as nn

from models.base_model import BaseModel
from utilities.data import AMASSBatch
from utilities.losses import l1_loss
from utilities.utils import get_dct_matrix

class GCNAttModel(BaseModel):

    def __init__(self,config):
        super(GCNAttModel, self).__init__(config)

    def create_model(self, in_features=135, kernel_size=17, d_model=256, num_stage=12, dct_n=34):
        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        # assert kernel_size == 10

        self.GCEncoder = nn.Sequential(GCN4attn.GraphConvolution(in_features=kernel_size,out_features=7,node_n=in_features),
                                    GCN4attn.GraphConvolution(in_features=7,out_features=5,node_n=in_features))
        self.GCDecoder = nn.Sequential(GCN4attn.GraphConvolution(in_features=5,out_features=7,node_n=in_features),
                                    GCN4attn.GraphConvolution(in_features=7,out_features=kernel_size,node_n=in_features))

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
        kl = self.kernel_size
        idx_val = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx_val].clone().reshape(
            [bs * vn, vl, -1])
        # print('src_value_tmp :' + str(src_value_tmp.shape))
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]
        # print('src_value_tmp :' + str(src_value_tmp.shape))

        idx_keys = np.expand_dims(np.arange(kl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        
        src_key_tmp = src_tmp[:,idx_keys].clone().reshape([bs*vn,kl,-1]).transpose(1,2)
        # print("src_key_tmp :"+str(src_key_tmp.shape))

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.GCEncoder(src_key_tmp)
        decoded_key_tmp = self.GCDecoder(key_tmp)
        key_tmp = key_tmp.reshape([bs,vn,135,-1]).detach()
        # print('key_tmp:'+str(key_tmp.shape))
        query_tmp = self.GCEncoder(src_query_tmp)
        decoded_query_tmp = self.GCDecoder(query_tmp)
        query_tmp = query_tmp.unsqueeze(dim=1).detach()
        # print('query_tmp:'+str(query_tmp.shape))
        
        score_tmp = torch.matmul(key_tmp, query_tmp.transpose(2,3)).diagonal(dim1=-2,dim2=-1) + 1e-15
        # print('score tmp'+str(score_tmp.shape))
        # print(key_tmp.norm(p=2,dim=-1).shape)
        # print(query_tmp.norm(p=2,dim=-1).shape)
        norms = torch.mul(key_tmp.norm(p=2,dim=-1),query_tmp.norm(p=2,dim=-1))
        att_tmp = torch.div(score_tmp,norms).mean(dim=-1)
        att_tmp = torch.div(att_tmp,att_tmp.sum(dim=-1).unsqueeze(dim=1)).unsqueeze(dim=1)
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
        model_out["loss_AE"] = (decoded_key_tmp - src_key_tmp).pow(2).reshape([bs,vn,135,-1]).sum(dim=-1).sum(dim=-1).mean(dim=1).mean(dim=0) +\
        (decoded_query_tmp - src_query_tmp).pow(2).squeeze().sum(dim=-1).sum(dim=-1).mean(dim=0)
        # print(model_out['loss_AE'].shape)
        return model_out

    def backward(self, batch: AMASSBatch, model_out):

        predictions = model_out["predictions"]
        targets = batch.poses[:, self.config.seed_seq_len :]

        total_loss = l1_loss(predictions, targets)
        # print(total_loss.shape)

        # If you have more than just one loss, just add them to this dict and they will
        # automatically be logged.
        loss_vals = {"total_loss": total_loss.cpu().item(), "AE_loss":model_out["loss_AE"].cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also
            # be called when evaluating the model on the validation set.
            total_loss.backward()
            model_out['loss_AE'].backward()

        return loss_vals, targets
