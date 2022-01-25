import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from ..modules import CompGCNConv, CompGCNConvBasis
from argparse import Namespace


class CompGCN(nn.Module):
    def __init__(self, edge_index, edge_type, entity_dict_len, relation_dict_len, embedding_dim, num_bases=-1):
        super(CompGCN, self).__init__()
        self.name = "CompGCN"

        args = Namespace(bias=True,
                         dropout=0.1,
                         opn="sub",
                         )
        self.p = args

        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

        self.edge_index = nn.Parameter(
            torch.LongTensor(edge_index).t(),
            requires_grad=False
        )
        self.edge_type = nn.Parameter(
            torch.LongTensor(edge_type),
            requires_grad=False
        )

        # edge_index = torch.LongTensor(edge_index).t()
        # edge_type = torch.LongTensor(edge_type)
        #
        # self.edge_index = edge_index
        # self.edge_type = edge_type
        self.num_bases = num_bases

        self.gcn_dim = embedding_dim
        self.init_dim = embedding_dim  # 暂时还不太清楚这两者有什么区别 先保持一致
        self.init_embed = get_param((entity_dict_len, self.init_dim))

        if self.num_bases > 0:
            self.init_rel = get_param((self.num_bases, self.init_dim))
        else:  # 先类似transe的写法
            self.init_rel = get_param((relation_dict_len, self.init_dim))

        if self.num_bases > 0:  # 先固定为一层的GCN网络
            self.conv1 = CompGCNConvBasis(self.init_dim, self.gcn_dim, relation_dict_len, self.num_bases, act=self.act,
                                          params=self.p)
        else:
            self.conv1 = CompGCNConv(self.init_dim, self.gcn_dim, relation_dict_len, act=self.act, params=self.p)
            self.drop1 = nn.Dropout(self.p.dropout)

        self.register_parameter('bias', Parameter(torch.zeros(entity_dict_len)))

    def forward(self, sample):
        batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]

        r = torch.cat([self.init_rel, -self.init_rel], dim=0)
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)

        head_embeddimg = torch.index_select(x, 0, batch_h)
        tail_embedding = torch.index_select(x, 0, batch_t)
        relation_embeddimg = torch.index_select(r, 0, batch_r)

        score = F.pairwise_distance(head_embeddimg + relation_embeddimg, tail_embedding, 2)
        return score


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


# class CompGCN(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim, gamma=6.0):
#         super(CompGCN, self).__init__()
#         self.name = "CompGCN"
#     self.embedding_dim = embedding_dim
#     self.entity_dict_len = entity_dict_len
#     self.relation_dict_len = relation_dict_len
#
#     self.entity_embedding = nn.Embedding(entity_dict_len, embedding_dim)
#     self.relation_embedding_head = nn.Embedding(relation_dict_len, embedding_dim)
#     self.relation_embedding_tail = nn.Embedding(relation_dict_len, embedding_dim)
#
#     self.gamma = nn.Parameter(
#         torch.Tensor([gamma]),
#         requires_grad=False
#     )
#
#     nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#     nn.init.xavier_uniform_(self.relation_embedding_head.weight.data)
#     nn.init.xavier_uniform_(self.relation_embedding_head.weight.data)
#
# def forward(self, sample):
#     batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
#
#     h = self.entity_embedding(batch_h)
#     r_h = self.relation_embedding_head(batch_r)
#     r_t = self.relation_embedding_tail(batch_r)
#     t = self.entity_embedding(batch_t)  # (batch,dim_entity)
#
#     # constraint is only added on entity embeddings
#     h = F.normalize(h, p=2.0, dim=-1)
#     t = F.normalize(t, p=2.0, dim=-1)
#
#     score = torch.norm(h * r_h - t * r_t, p=1, dim=-1)  # (batch,)
#     return self.gamma.item() - score
#
# def get_score(self, sample):
#     return self.forward(sample)
