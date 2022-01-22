import torch
import torch.nn as nn
import torch.nn.functional as F
from ...adapter import *


class TransE_Adapter(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50,p=1):
        super(TransE_Adapter, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.p = p
        self.name = "TransE"

        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def get_score(self,triplet_idx):
        output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)

    def forward(self,triplet_idx):
        return self.get_score(triplet_idx)

    @description
    @graph
    @nodetype
    @time
    def get_embedding(self,triplet_idx):
        head_embedding = self.entity_embedding(triplet_idx[:, 0])
        relation_embedding = self.relation_embedding(triplet_idx[:, 1])
        tail_embedding = self.entity_embedding(triplet_idx[:, 2])
        return head_embedding,relation_embedding,tail_embedding

    def _forward(self, triplet_idx):
        head_embedding,relation_embedding,tail_embedding=self.get_embedding(triplet_idx)

        head_embedding = F.normalize(head_embedding, p=2, dim=1)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=1)

        head_embedding = torch.unsqueeze(head_embedding , 1)
        relation_embedding = torch.unsqueeze(relation_embedding , 1)
        tail_embedding = torch.unsqueeze(tail_embedding, 1)
        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding

        return output # (batch,3,embedding_dim)

