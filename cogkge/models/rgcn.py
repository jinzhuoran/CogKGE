import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from ..modules import RGCNConv
from argparse import Namespace
from cogkge.models.basemodel import BaseModel

class RGCN(BaseModel):
    def __init__(self, edge_index, edge_type, entity_dict_len,relation_dict_len, embedding_dim):
        super(RGCN, self).__init__(model_name="RGCN")
        args = Namespace(bias=False,
                         dropout=0.1,
                         b_norm=False,
                         )
        self.p = args
        self.act = torch.tanh

        self.edge_index = nn.Parameter(
            torch.LongTensor(edge_index).t(),
            requires_grad=False
        )
        self.edge_type = nn.Parameter(
            torch.LongTensor(edge_type),
            requires_grad=False
        )

        self.init_dim = embedding_dim
        self.init_embed = self.get_param((entity_dict_len, self.init_dim))
        self.init_rel = self.get_param((relation_dict_len,self.init_dim))

        self.conv1 = RGCNConv(self.init_dim, self.init_dim, relation_dict_len,act=self.act,params=self.p)
        self.linear = nn.Linear(3 * embedding_dim,entity_dict_len)

    def set_model_config(self,model_loss=None,model_metric=None,model_negative_sampler=None,model_device=None):
        #设置模型使用的metric和loss
        self.model_loss=model_loss
        self.model_metric=model_metric
        self.model_negative_sampler=model_negative_sampler
        self.model_device=model_device

    def loss(self,data):
        batch_h, batch_r, batch_t,batch_label = self.get_batch(data)
        output = self.forward(batch_h,batch_r,batch_t,batch_label)
        return self.model_loss(output,batch_label)



    def forward(self,batch_h,batch_r,batch_t,label):
        x = self.conv1(self.init_embed,self.edge_index)

        head_embedding = torch.index_select(x, 0, batch_h)
        tail_embedding = torch.index_select(x, 0, batch_t)
        relation_embedding = torch.index_select(self.init_rel,0,batch_r)

        output = F.relu(torch.cat([head_embedding, tail_embedding, relation_embedding], dim=1))
        return torch.sigmoid(self.linear(output))


    def get_param(self, shape):
        param = Parameter(torch.Tensor(*shape))
        xavier_normal_(param.data)
        return param

    def get_batch(self,data):
        h=data["h"].to(self.model_device)
        r=data["r"].to(self.model_device)
        t=data["t"].to(self.model_device)
        label = data["label"].to(self.model_device)
        return h,r,t,label