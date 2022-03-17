import torch
import numpy as np
from torch.nn.init import xavier_normal_
from cogkge.models.basemodel import BaseModel

class TuckER(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 d1,
                 d2,
                 input_dropout=0.3,
                 hidden_dropout1=0.4,
                 hidden_dropout2=0.5):
        super().__init__(model_name="TuckER")

        self.E = torch.nn.Embedding(entity_dict_len, d1)
        self.R = torch.nn.Embedding(relation_dict_len, d2)
        # self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
        #                                          dtype=torch.float, device="cuda", requires_grad=True))
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, h_r_true):
        e1_idx=h_r_true[:,0]
        r_idx=h_r_true[:,1]
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def get_batch(self,data):
        h = data[0].to(self.model_device)
        r = data[1].to(self.model_device)
        label = data[2].to(self.model_device)
        return h,r,label

    def loss(self,data):
        h,r,label = self.get_batch(data)
        pred = self.forward(torch.cat([h.unsqueeze(1),r.unsqueeze(1)],dim=1))
        return self.model_loss(pred,label)
