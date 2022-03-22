import torch
import torch.nn as nn
import torch.nn.functional as F

from cogkge.models.basemodel import BaseModel


class Rescal(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 embedding_dim,
                 penalty_weight=0.0):
        super().__init__(model_name="Rescal", penalty_weight=penalty_weight)
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.entity_embedding = nn.Embedding(entity_dict_len, embedding_dim)
        self.relation_embedding = nn.Embedding(relation_dict_len, embedding_dim * embedding_dim)

        self._reset_param()

    def _reset_param(self):
        # 重置参数
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def get_realation_embedding(self, relation_ids):
        # 得到关系的embedding
        return self.r_embedding(relation_ids)

    def get_entity_embedding(self, entity_ids):
        # 得到实体的embedding
        return self.e_embedding(entity_ids)
    def get_triplet_embedding(self, data):
        # 得到三元组的embedding
        h_embedding = self.e_embedding(data[0])
        r_embedding = self.r_embedding(data[1])
        t_embedding = self.e_embedding(data[2])
        return h_embedding, r_embedding, t_embedding

    def forward(self, data):
        batch_h, batch_r, batch_t = data[:, 0], data[:, 1], data[:, 2]
        A = self.entity_embedding(batch_h)  # (batch,embedding)
        A = F.normalize(A, p=2, dim=-1)
        R = self.relation_embedding(batch_r).view(-1, self.embedding_dim,self.embedding_dim)  # (batch,embedding,embedding)
        A_T = self.entity_embedding(batch_t).view(-1, self.embedding_dim, 1)  # (batch,embedding,1)
        A_T = F.normalize(A_T, p=2, dim=1)

        tr = torch.matmul(R, A_T)  # (batch,embedding_dim,1)
        tr = tr.view(-1, self.embedding_dim)  # (batch,embedding_dim)

        return -torch.sum(A * tr, dim=-1)  # (batch,)

    def loss(self, data):
        # 计算损失
        pos_data = data
        pos_data = self.data_to_device(pos_data)
        neg_data = self.model_negative_sampler.create_negative(data)
        neg_data = self.data_to_device(neg_data)

        pos_score = self.forward(pos_data)
        neg_score = self.forward(neg_data)

        return self.model_loss(pos_score, neg_score) + self.penalty(data)

