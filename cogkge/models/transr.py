# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class TransR(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, dim_entity, dim_relation, p=2.0):
#         super(TransR, self).__init__()
#         self.name = "TransR"
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.dim_entity = dim_entity
#         self.dim_relation = dim_relation
#         self.p = p
#
#         self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
#         self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
#         self.transfer_matrix = nn.Embedding(relation_dict_len, dim_entity * dim_relation)
#
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#         nn.init.xavier_uniform_(self.transfer_matrix.weight.data)
#
#     def transfer(self, e, r_transfer):
#         r_transfer = r_transfer.view(-1, self.dim_entity, self.dim_relation)
#         e = torch.unsqueeze(e, 1)
#
#         return torch.squeeze(torch.bmm(e, r_transfer))
#
#     def get_score(self, sample):
#         output = self._forward(sample)
#         score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
#         return score  # (batch,)
#
#     def get_embedding(self, sample):
#         return self._forward(sample)
#
#     def forward(self, sample):
#         return self.get_score(sample)
#
#     def _forward(self, sample):  # sample:(batch,3)
#         batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
#         h = self.entity_embedding(batch_h)
#         r = self.relation_embedding(batch_r)
#         t = self.entity_embedding(batch_t)
#
#         r_transfer = self.transfer_matrix(batch_r)
#
#         h = F.normalize(h, p=2.0, dim=-1)
#         t = F.normalize(t, p=2.0, dim=-1)  # ||h|| <= 1  ||t|| <= 1
#         r = F.normalize(r, p=2.0, dim=-1)
#
#         h = self.transfer(h, r_transfer)
#         t = self.transfer(t, r_transfer)
#
#         h = F.normalize(h, p=2.0, dim=-1)
#         r = F.normalize(r, p=2.0, dim=-1)
#         t = F.normalize(t, p=2.0, dim=-1)
#
#         h = torch.unsqueeze(h, 1)
#         r = torch.unsqueeze(r, 1)
#         t = torch.unsqueeze(t, 1)
#
#         return torch.cat((h, r, t), dim=1)

import torch.nn as nn
import torch.nn.functional as F
import torch
from cogkge.models.basemodel import BaseModel
from cogkge.adapter import *


class TransR(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 entity_embedding_dim=50,
                 relation_embedding_dim=50,
                 p_norm=1,
                 penalty_weight=0.0):
        super().__init__(model_name="TransR", penalty_weight=penalty_weight)
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim=relation_embedding_dim
        self.p_norm = p_norm

        self.e_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.entity_embedding_dim)
        self.r_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.relation_embedding_dim)
        self.matrix=nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.entity_embedding_dim*self.relation_embedding_dim)

        self._reset_param()

    def _reset_param(self):
        # 重置参数
        nn.init.xavier_uniform_(self.e_embedding.weight.data)
        nn.init.xavier_uniform_(self.r_embedding.weight.data)
        nn.init.xavier_uniform_(self.matrix.weight.data)

    def forward(self, data):
        # 前向传播
        r=data[1]
        h_embedding, r_embedding, t_embedding = self.get_triplet_embedding(data=data)
        r_matrix=self.matrix(r).reshape((-1,self.entity_embedding_dim,self.relation_embedding_dim))
        h_embedding=torch.matmul(h_embedding.unsqueeze(1),r_matrix).squeeze(1)
        t_embedding=torch.matmul(t_embedding.unsqueeze(1),r_matrix).squeeze(1)



        h_embedding = F.normalize(h_embedding, p=2, dim=1)
        r_embedding = F.normalize(r_embedding, p=2, dim=1)
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
        pos_data=data
        pos_data= self.data_to_device(pos_data)
        neg_data=self.model_negative_sampler.create_negative(data)
        neg_data = self.data_to_device(neg_data)

        pos_score = self.forward(pos_data)
        neg_score = self.forward(neg_data)

        return self.model_loss(pos_score, neg_score) + self.penalty(data)


