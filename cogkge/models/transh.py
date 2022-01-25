import torch
import torch.nn as nn
import torch.nn.functional as F


class TransH(nn.Module):

    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50, p=2, epsilon=0.01):
        super(TransH, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.dim = embedding_dim
        self.p = p
        self.epsilon = epsilon
        self.name = "TransH"

        self.entity_embedding = nn.Embedding(self.entity_dict_len, self.dim)
        self.relation_embedding = nn.Embedding(self.relation_dict_len, self.dim)
        self.w_vector = nn.Embedding(self.relation_dict_len, self.dim)

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.w_vector.weight.data)

        self.head_batch_embedding = None
        self.relation_batch_embedding = None
        self.tail_batch_embedding = None
        self.w_r_batch_embedding = None

    def get_score(self, triplet_idx):
        # (batch,3)

        output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)

        return score

    def get_embedding(self, triplet_idx):
        return self._forward(triplet_idx)

    def forward(self, triplet_idx):
        return self.get_score(triplet_idx)

    def _forward(self, triplet_idx):
        # (batch,d)
        head_embedding = self.entity_embedding(triplet_idx[:, 0])
        relation_embedding = self.relation_embedding(triplet_idx[:, 1])
        tail_embedding = self.entity_embedding(triplet_idx[:, 2])
        w_r = self.w_vector(triplet_idx[:, 1])  # (batch,d)
        w_r = F.normalize(w_r, p=2, dim=1)

        self.head_batch_embedding = head_embedding
        self.relation_batch_embedding = relation_embedding
        self.tail_batch_embedding = tail_embedding
        self.w_r_batch_embedding = w_r

        head_embedding = head_embedding - torch.sum(head_embedding * w_r, 1, True) * w_r
        tail_embedding = tail_embedding - torch.sum(tail_embedding * w_r, 1, True) * w_r

        head_embedding = torch.unsqueeze(head_embedding, 1)
        relation_embedding = torch.unsqueeze(relation_embedding, 1)
        tail_embedding = torch.unsqueeze(tail_embedding, 1)
        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding

        return output

    def get_penalty(self):
        penalty = (
                          torch.mean(self.head_batch_embedding ** 2) +
                          torch.mean(self.relation_batch_embedding ** 2) +
                          torch.mean(self.tail_batch_embedding ** 2)
                  ) / 3
        return penalty