import torch
import torch.nn as nn

from .basemodel import BaseModel

class DistMult(BaseModel):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,penalty_weight=0.0):
        super(DistMult, self).__init__(model_name="DistMult",penalty_weight=penalty_weight)
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.entity_embedding = nn.Embedding(self.entity_dict_len, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.relation_dict_len, self.embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

        self.head_batch_embedding = None
        self.relation_batch_embedding = None
        self.tail_batch_embedding = None

    def forward(self,data):
        h,r,t = data[0],data[1],data[2]

        head_embedding = self.entity_embedding(h)
        relation_embedding = self.relation_embedding(r)
        tail_embedding = self.entity_embedding(t)

        self.head_batch_embedding = head_embedding
        self.relation_batch_embedding = relation_embedding
        self.tail_batch_embedding = tail_embedding

        score = head_embedding * relation_embedding * tail_embedding

        score = torch.sum(score, -1).flatten()
        return score

    def loss(self, data):
        pos_data= self.data_to_device(data)
        neg_data=self.model_negative_sampler.create_negative(data)
        neg_data = self.data_to_device(neg_data)

        pos_score = self.forward(pos_data)
        neg_score = self.forward(neg_data)

        return self.model_loss(pos_score, neg_score) + self.penalty_weight * self.get_penalty()

    def get_penalty(self):
        penalty = (
                          torch.mean(self.head_batch_embedding ** 2) +
                          torch.mean(self.relation_batch_embedding ** 2) +
                          torch.mean(self.tail_batch_embedding ** 2)
                  ) / 3
        return penalty

# class DistMult(torch.nn.Module):
#
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim):
#         super(DistMult, self).__init__()
#         self.name = "DistMult"
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.embedding_dim = embedding_dim
#         self.entity_embedding = nn.Embedding(self.entity_dict_len, self.embedding_dim)
#         self.relation_embedding = nn.Embedding(self.relation_dict_len, self.embedding_dim)
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#
#         self.head_batch_embedding = None
#         self.relation_batch_embedding = None
#         self.tail_batch_embedding = None
#
#     def get_score(self, triplet_idx):
#         h, r, t = self.get_embedding(triplet_idx)
#         h = h.view(-1, r.shape[0], h.shape[-1])
#         t = t.view(-1, r.shape[0], t.shape[-1])
#         r = r.view(-1, r.shape[0], r.shape[-1])
#         score = h * (r * t)
#         score = torch.sum(score, -1).flatten()
#         return score
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
#         self.head_batch_embedding = head_embedding
#         self.relation_batch_embedding = relation_embedding
#         self.tail_batch_embedding = tail_embedding
#
#         return head_embedding, relation_embedding, tail_embedding
#
#     def get_penalty(self):
#         penalty = (
#                           torch.mean(self.head_batch_embedding ** 2) +
#                           torch.mean(self.relation_batch_embedding ** 2) +
#                           torch.mean(self.tail_batch_embedding ** 2)
#                   ) / 3
#         return penalty
