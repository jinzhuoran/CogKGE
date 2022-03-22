import numpy as np
from torch.nn.init import xavier_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogkge.models.basemodel import BaseModel

class TuckER(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 d1,
                 d2,
                 input_dropout=0.2,
                 hidden_dropout1=0.2,
                 hidden_dropout2=0.3,
                 penalty_weight=0.0):
        super().__init__(model_name="TuckER",penalty_weight=penalty_weight)
        self.entity_dict_len=entity_dict_len
        self.relation_dict_len=relation_dict_len
        self.d1=d1
        self.d2=d2
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout2)

        self.e_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.d1)
        self.r_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.d2)


        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.d2, self.d1, self.d1)),dtype=torch.float, requires_grad=True))
        self.bn0 = torch.nn.BatchNorm1d(self.d1)
        self.bn1 = torch.nn.BatchNorm1d(self.d1)

        self._reset_param()

    def _reset_param(self):
        # 重置参数
        nn.init.xavier_normal_(self.e_embedding.weight.data)
        nn.init.xavier_normal_(self.r_embedding.weight.data)

    def forward(self, h_r_true):
        e1_idx = h_r_true[:, 0]
        r_idx = h_r_true[:, 1]
        e1 = self.e_embedding (e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.r_embedding(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.e_embedding.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def get_batch(self, data):
        h = data[0]
        r = data[1]
        label = data[2]
        return h, r, label

    def loss(self, data):
        data=self.data_to_device(data)
        h, r, label = self.get_batch(data)
        pred = self.forward(torch.cat([h.unsqueeze(1), r.unsqueeze(1)], dim=1))
        return self.model_loss(pred, label)+ self.penalty(data)

