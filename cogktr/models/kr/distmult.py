import torch
import torch.nn as nn
import torch.nn.functional as F


# class DistMult(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50):
#         super(DistMult, self).__init__()
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.embedding_dim = embedding_dim
#         self.name = "DistMult"
#
#         self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
#         self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#
#     def get_score(self,triplet_idx):
#         output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
#         score = output[:, 0] * output[:, 1] * output[:, 2]
#         return score  # (batch,)
#
#     def forward(self,triplet_idx):
#         return self.get_score(triplet_idx)
#
#
#     def get_embedding(self,triplet_idx):
#         return self._forward(triplet_idx)
#
#
#     def _forward(self, triplet_idx):
#         head_embedding = self.entity_embedding(triplet_idx[:, 0])
#         relation_embedding = self.relation_embedding(triplet_idx[:, 1])
#         tail_embedding = self.entity_embedding(triplet_idx[:, 2])
#
#         head_embedding = F.normalize(head_embedding, p=2, dim=1)
#         tail_embedding = F.normalize(tail_embedding, p=2, dim=1)
#
#         head_embedding = torch.unsqueeze(head_embedding , 1)
#         relation_embedding = torch.unsqueeze(relation_embedding , 1)
#         tail_embedding = torch.unsqueeze(tail_embedding, 1)
#         triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)
#
#         output = triplet_embedding
#
#         return output # (batch,3,embedding_dim)

class DistMult(torch.nn.Module):

    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim):
        super(DistMult, self).__init__()
        self.name = "DistMult"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.entity_embedding = nn.Embedding(self.entity_dict_len, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.relation_dict_len, self.embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

        self.head_batch_embedding=None
        self.relation_batch_embedding=None
        self.tail_batch_embedding = None

    def get_score(self, triplet_idx):
        h, r, t = self.get_embedding(triplet_idx)
        h = h.view(-1, r.shape[0], h.shape[-1])
        t = t.view(-1, r.shape[0], t.shape[-1])
        r = r.view(-1, r.shape[0], r.shape[-1])
        score = h * (r * t)
        score = torch.sum(score, -1).flatten()
        return score

    def forward(self, triplet_idx):
        return self.get_score(triplet_idx)

    def get_embedding(self, triplet_idx):
        return self._forward(triplet_idx)

    def _forward(self, triplet_idx):
        head_embedding = self.entity_embedding(triplet_idx[:, 0])
        relation_embedding = self.relation_embedding(triplet_idx[:, 1])
        tail_embedding = self.entity_embedding(triplet_idx[:, 2])

        self.head_batch_embedding = head_embedding
        self.relation_batch_embedding = relation_embedding
        self.tail_batch_embedding = tail_embedding

        return head_embedding, relation_embedding, tail_embedding
    def get_penalty(self):
        # constraint_1=torch.sum(nn.ReLU(inplace=False)(self.head_batch_embedding ** 2-1/len(self.head_batch_embedding)))
        # constraint_2=torch.sum(nn.ReLU(inplace=False)(self.tail_batch_embedding ** 2-1/len(self.tail_batch_embedding)))
        # constraint_3=sum(sum((self.relation_batch_embedding*self.w_r_batch_embedding)** 2)/sum(self.relation_batch_embedding**2)-self.epsilon**2)
        # penalty=constraint_1+constraint_2+constraint_3
        penalty = (
                          torch.mean(self.head_batch_embedding ** 2) +
                   torch.mean(self.relation_batch_embedding ** 2) +
                   torch.mean(self.tail_batch_embedding ** 2)
                   # torch.mean(self.w_r_batch_embedding ** 2)
                  ) / 3
        return penalty
