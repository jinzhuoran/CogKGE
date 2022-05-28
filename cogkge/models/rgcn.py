import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from ..modules import RGCNConv
from argparse import Namespace
from cogkge.models.basemodel import BaseModel


# core based
class RGCN(BaseModel):
    def __init__(self, edge_index, edge_type, entity_dict_len, relation_dict_len, embedding_dim):
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
        self.init_rel = self.get_param((relation_dict_len, self.init_dim))

        self.conv1 = RGCNConv(self.init_dim, self.init_dim, relation_dict_len, act=self.act, params=self.p)
        # self.linear = nn.Linear(2 * embedding_dim, entity_dict_len)

        self.empty = nn.Embedding(1, 2)

    def loss(self, data):
        pos_data= self.data_to_device(data)
        neg_data=self.model_negative_sampler.create_negative(data)
        neg_data = self.data_to_device(neg_data)

        pos_score = self.forward(pos_data)
        neg_score = self.forward(neg_data)

        return self.model_loss(pos_score, neg_score)

    def forward(self, sample):
        # batch_h,batch_r,batch_label = data_batch
        batch_h,batch_r,batch_t = sample[0],sample[1],sample[2]
        x = self.conv1(self.init_embed, self.edge_index)

        head_embedding = torch.index_select(x, 0, batch_h)
        tail_embedding = torch.index_select(x, 0, batch_t)
        relation_embedding = torch.index_select(self.init_rel, 0, batch_r)

        score = F.pairwise_distance(head_embedding + relation_embedding, tail_embedding, 2)
        return score

    def get_param(self, shape):
        param = Parameter(torch.Tensor(*shape))
        xavier_normal_(param.data)
        return param

# # classification based
# class RGCN(BaseModel):
#     def __init__(self, edge_index, edge_type, entity_dict_len,relation_dict_len, embedding_dim):
#         super(RGCN, self).__init__(model_name="RGCN")
#         args = Namespace(bias=False,
#                          dropout=0.1,
#                          b_norm=False,
#                          )
#         self.p = args
#         self.act = torch.tanh
#
#         self.edge_index = nn.Parameter(
#             torch.LongTensor(edge_index).t(),
#             requires_grad=False
#         )
#         self.edge_type = nn.Parameter(
#             torch.LongTensor(edge_type),
#             requires_grad=False
#         )
#
#         self.init_dim = embedding_dim
#         self.init_embed = self.get_param((entity_dict_len, self.init_dim))
#         self.init_rel = self.get_param((relation_dict_len,self.init_dim))
#
#         self.conv1 = RGCNConv(self.init_dim, self.init_dim, relation_dict_len,act=self.act,params=self.p)
#         self.linear = nn.Linear(2 * embedding_dim,entity_dict_len)
#
#
#     def loss(self,data):
#         data = self.data_to_device(data)
#         h,r,batch_label = data
#         # data_batch = torch.cat(data)
#         output = self.forward(torch.cat([h.unsqueeze(1), r.unsqueeze(1)], dim=1))
#         return self.model_loss(output,batch_label)
#
#
#     def forward(self,data_batch):
#         # batch_h,batch_r,batch_label = data_batch
#         batch_h,batch_r  = data_batch[:,0],data_batch[:,1]
#         x = self.conv1(self.init_embed,self.edge_index)
#
#         head_embedding = torch.index_select(x, 0, batch_h)
#         # tail_embedding = torch.index_select(x, 0, batch_t)
#         # tail_embedding = torch.matmul(batch_label,x)
#         relation_embedding = torch.index_select(self.init_rel,0,batch_r)
#
#         output = F.relu(torch.cat([head_embedding, relation_embedding], dim=1))
#         return torch.sigmoid(self.linear(output))
#
#
#     def get_param(self, shape):
#         param = Parameter(torch.Tensor(*shape))
#         xavier_normal_(param.data)
#         return param
#