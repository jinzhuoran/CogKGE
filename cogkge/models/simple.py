# import torch
# import torch.nn as nn
#
#
# class SimplE(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim):
#         super(SimplE, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.name = "SimplE"
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.head_embedding = nn.Embedding(entity_dict_len, embedding_dim)
#         self.tail_embedding = nn.Embedding(entity_dict_len, embedding_dim)
#         self.relation_embedding = nn.Embedding(relation_dict_len, embedding_dim)
#         self.relation_inverse_embedding = nn.Embedding(relation_dict_len, embedding_dim)
#         self.r = None
#         self.r_ = None
#         self.h = None
#         self.h_ = None
#         self.t = None
#         self.t_ = None
#
#         nn.init.xavier_uniform_(self.head_embedding.weight.data)
#         nn.init.xavier_uniform_(self.tail_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_inverse_embedding.weight.data)
#
#     def forward(self, sample):
#         batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
#         h = self.head_embedding(batch_h)
#         r = self.relation_embedding(batch_r)
#         self.r = r
#         t = self.tail_embedding(batch_t)
#
#         self.h = h
#         self.t = t
#         score_front = torch.sum(h * r * t, dim=1)  # (batch,)
#
#         h_ = self.head_embedding(batch_t)
#         r_ = self.relation_inverse_embedding(batch_r)
#         self.r_ = r_
#         t_ = self.tail_embedding(batch_h)
#
#         self.h_ = h_
#         self.t_ = t_
#         score_reverse = torch.sum(h_ * r_ * t_, dim=1)  # (batch,)
#
#         return (score_front + score_reverse) / 2
#
#     def get_score(self, sample):
#         return self.forward(sample)
#
#     def get_penalty(self):
#         return torch.norm(torch.mean(self.r + self.r_ + self.t + self.t_ + self.h + self.h_))
import torch
import torch.nn as nn

from cogkge.models.basemodel import BaseModel


class SimplE(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 embedding_dim,
                 penalty_weight=0.0):
        super().__init__(model_name="SimplE", penalty_weight=penalty_weight)
        super(SimplE, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim=embedding_dim
        self.h_embedding = nn.Embedding(entity_dict_len, embedding_dim)
        self.r_embedding = nn.Embedding(relation_dict_len, embedding_dim)
        self.t_embedding = nn.Embedding(entity_dict_len, embedding_dim)
        self.r_inverse_embedding = nn.Embedding(relation_dict_len, embedding_dim)

        self._reset_param()

    def _reset_param(self):
        # 重置参数
        nn.init.xavier_uniform_(self.h_embedding.weight.data)
        nn.init.xavier_uniform_(self.r_embedding.weight.data)
        nn.init.xavier_uniform_(self.t_embedding.weight.data)
        nn.init.xavier_uniform_(self.r_inverse_embedding.weight.data)

    def forward(self, data):
        batch_h, batch_r, batch_t = data[0], data[1], data[2]
        h = self.h_embedding(batch_h)
        r = self.r_embedding(batch_r)
        t = self.t_embedding(batch_t)
        score_front = torch.sum(h * r * t, dim=1)  # (batch,)

        h_ = self.h_embedding(batch_t)
        r_ = self.r_inverse_embedding(batch_r)
        t_ = self.t_embedding(batch_h)
        score_reverse = torch.sum(h_ * r_ * t_, dim=1)  # (batch,)

        return (score_front + score_reverse) / 2

    def penalty(self,data):
        batch_h, batch_r, batch_t = data[0], data[1], data[2]
        h = self.h_embedding(batch_h)
        r = self.r_embedding(batch_r)
        t = self.t_embedding(batch_t)
        h_ = self.h_embedding(batch_t)
        r_ = self.r_inverse_embedding(batch_r)
        t_ = self.t_embedding(batch_h)
        return torch.norm(torch.mean(r + r_ + t + t_ + h + h_))

    def loss(self, data):
        # 计算损失
        pos_data = data
        pos_data = self.data_to_device(pos_data)
        neg_data = self.model_negative_sampler.create_negative(data)
        neg_data = self.data_to_device(neg_data)

        pos_score = self.forward(pos_data)
        neg_score = self.forward(neg_data)

        return self.model_loss(pos_score, neg_score) + self.penalty_weight*self.penalty(data)
