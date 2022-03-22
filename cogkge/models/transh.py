# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class TransH(nn.Module):
#
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50, p=2, epsilon=0.01):
#         super(TransH, self).__init__()
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.dim = embedding_dim
#         self.p = p
#         self.epsilon = epsilon
#         self.name = "TransH"
#
#         self.entity_embedding = nn.Embedding(self.entity_dict_len, self.dim)
#         self.relation_embedding = nn.Embedding(self.relation_dict_len, self.dim)
#         self.w_vector = nn.Embedding(self.relation_dict_len, self.dim)
#
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#         nn.init.xavier_uniform_(self.w_vector.weight.data)
#
#         self.head_batch_embedding = None
#         self.relation_batch_embedding = None
#         self.tail_batch_embedding = None
#         self.w_r_batch_embedding = None
#
#     def get_score(self, triplet_idx):
#         # (batch,3)
#
#         output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
#         score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
#
#         return score
#
#     def get_embedding(self, triplet_idx):
#         return self._forward(triplet_idx)
#
#     def forward(self, triplet_idx):
#         return self.get_score(triplet_idx)
#
#     def _forward(self, triplet_idx):
#         # (batch,d)
#         head_embedding = self.entity_embedding(triplet_idx[:, 0])
#         relation_embedding = self.relation_embedding(triplet_idx[:, 1])
#         tail_embedding = self.entity_embedding(triplet_idx[:, 2])
#         w_r = self.w_vector(triplet_idx[:, 1])  # (batch,d)
#         w_r = F.normalize(w_r, p=2, dim=1)
#
#         self.head_batch_embedding = head_embedding
#         self.relation_batch_embedding = relation_embedding
#         self.tail_batch_embedding = tail_embedding
#         self.w_r_batch_embedding = w_r
#
#         head_embedding = head_embedding - torch.sum(head_embedding * w_r, 1, True) * w_r
#         tail_embedding = tail_embedding - torch.sum(tail_embedding * w_r, 1, True) * w_r
#
#         head_embedding = torch.unsqueeze(head_embedding, 1)
#         relation_embedding = torch.unsqueeze(relation_embedding, 1)
#         tail_embedding = torch.unsqueeze(tail_embedding, 1)
#         triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)
#
#         output = triplet_embedding
#
#         return output
#
#     def get_penalty(self):
#         penalty = (
#                           torch.mean(self.head_batch_embedding ** 2) +
#                           torch.mean(self.relation_batch_embedding ** 2) +
#                           torch.mean(self.tail_batch_embedding ** 2)
#                   ) / 3
#         return


import torch.nn as nn
import torch.nn.functional as F
import torch
from cogkge.models.basemodel import BaseModel
from cogkge.adapter import *


class TransH(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 embedding_dim=50,
                 p_norm=1,
                 penalty_weight=0.0):
        super().__init__(model_name="TransH", penalty_weight=penalty_weight)
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.p_norm = p_norm

        self.e_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.r_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.w = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim )

        self._reset_param()

    def _reset_param(self):
        # 重置参数
        nn.init.xavier_uniform_(self.e_embedding.weight.data)
        nn.init.xavier_uniform_(self.r_embedding.weight.data)
        nn.init.xavier_uniform_(self.w.weight.data)

    def forward(self, data):
        # 前向传播
        r=data[1]
        h_embedding, r_embedding, t_embedding = self.get_triplet_embedding(data=data)
        w=self.w(r)

        h_embedding = F.normalize(h_embedding, p=2, dim=1)
        t_embedding = F.normalize(t_embedding, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        h_embedding=h_embedding-torch.sum(h_embedding*w,dim=1,keepdim=True)*h_embedding
        t_embedding=t_embedding-torch.sum(t_embedding*w,dim=1,keepdim=True)*t_embedding

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
        pos_data=data
        pos_data= self.data_to_device(pos_data)
        neg_data=self.model_negative_sampler.create_negative(data)
        neg_data = self.data_to_device(neg_data)

        pos_score = self.forward(pos_data)
        neg_score = self.forward(neg_data)

        return self.model_loss(pos_score, neg_score) + self.penalty(data)

    def penalty(self,data):
        batch_h, batch_r, batch_t = data[0], data[1], data[2]
        h = self.h_embedding(batch_h)
        r = self.r_embedding(batch_r)
        t = self.t_embedding(batch_t)
        w= self.w (batch_r)
        penalty=(torch.mean(h ** 2) +torch.mean(t ** 2) +torch.mean(r ** 2) +torch.mean(w ** 2)) / 4
        return penalty

