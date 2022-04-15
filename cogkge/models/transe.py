# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class TransE(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50, p=1):
#         super(TransE, self).__init__()
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.embedding_dim = embedding_dim
#         self.p = p
#         self.name = "TransE"
#
#         self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
#         self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#
#     def get_score(self, triplet_idx):
#         output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
#         score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
#         return score  # (batch,)
#
#     def forward(self, triplet_idx):
#         return self.get_score(triplet_idx)
#
#     def get_embedding(self, triplet_idx):
#         return self._forward(triplet_idx)
#
#     def _forward(self, triplet_idx):
#         head_embedding = self.entity_embedding(triplet_idx[:, 0])
#         relation_embedding = self.relation_embedding(triplet_idx[:, 1])
#         tail_embedding = self.entity_embedding(triplet_idx[:, 2])
#
#         head_embedding = F.normalize(head_embedding, p=2, dim=1)
#         tail_embedding = F.normalize(tail_embedding, p=2, dim=1)
#
#         head_embedding = torch.unsqueeze(head_embedding, 1)
#         relation_embedding = torch.unsqueeze(relation_embedding, 1)
#         tail_embedding = torch.unsqueeze(tail_embedding, 1)
#         triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)
#
#         output = triplet_embedding
#
#         return output  # (batch,3,embedding_dim)

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogkge.models.basemodel import BaseModel
from ..adapter import graph_adapter
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from ..modules import RGCNConv
from argparse import Namespace


class TransE(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 embedding_dim=50,
                 p_norm=1,
                 penalty_weight=0.0,
                 edge_index=None,
                 edge_type=None):
        super().__init__(model_name="TransE", penalty_weight=penalty_weight)
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.p_norm = p_norm
        self.edge_index = edge_type
        self.edge_type = edge_type

        if self.edge_index is not None and self.edge_type is not None:
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

        self.e_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.r_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)

        self._reset_param()

    def _reset_param(self):
        # 重置参数
        nn.init.xavier_uniform_(self.e_embedding.weight.data)
        nn.init.xavier_uniform_(self.r_embedding.weight.data)

    def forward(self, data):
        # 前向传播
        h_embedding, r_embedding, t_embedding = self.get_triplet_embedding(data=data)

        h_embedding = F.normalize(h_embedding, p=2, dim=1)
        t_embedding = F.normalize(t_embedding, p=2, dim=1)

        score = F.pairwise_distance(h_embedding + r_embedding, t_embedding, p=self.p_norm)

        return score

    def get_realation_embedding(self, relation_ids):
        # 得到关系的embedding
        return self.r_embedding(relation_ids)

    def get_entity_embedding(self, entity_ids):
        # 得到实体的embedding
        return self.e_embedding(entity_ids)

    # @description_adapter
    # @graph_adapter
    # @type_adapter
    # @time_adapter

    def get_triplet_embedding(self, data):
        # 得到三元组的embedding
        h_embedding = self.e_embedding(data[0])
        r_embedding = self.r_embedding(data[1])
        t_embedding = self.e_embedding(data[2])
        return h_embedding, r_embedding, t_embedding


    def loss(self, data):
        # 计算损失
        pos_data = data
        pos_data = self.data_to_device(pos_data)
        neg_data = self.model_negative_sampler.create_negative(data)
        neg_data = self.data_to_device(neg_data)

        pos_score = self.forward(pos_data)
        neg_score = self.forward(neg_data)

        return self.model_loss(pos_score, neg_score) + self.penalty(data)

    def get_param(self, shape):
        param = Parameter(torch.Tensor(*shape))
        xavier_normal_(param.data)
        return param