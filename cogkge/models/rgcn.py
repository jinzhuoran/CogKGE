import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from ..modules import RGCNConv
from argparse import Namespace

class RGCN(nn.Module):
    def __init__(self, edge_index, edge_type, entity_dict_len,relation_dict_len, embedding_dim):
        super(RGCN, self).__init__()
        self.name = "RGCN"

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



    def forward(self,sample):
        batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]

        x = self.conv1(self.init_embed,self.edge_index)

        head_embeddimg = torch.index_select(x, 0, batch_h)
        tail_embedding = torch.index_select(x, 0, batch_t)
        relation_embedding = torch.index_select(self.init_rel,0,batch_r)

        # DisMult
        # print(head_embeddimg.shape,self.init_rel.shape,tail_embedding.shape)
        score = head_embeddimg * (relation_embedding * tail_embedding)

        return score

    def get_param(self, shape):
        param = Parameter(torch.Tensor(*shape))
        xavier_normal_(param.data)
        return param