import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatE(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim):
        super(RotatE, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.name = "RotatE"
        self.square = embedding_dim ** 0.5
        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)

    def get_score(self,triplet_idx):
        output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] * output[:, 1], output[:, 2], p=2)
        return score  # (batch,) 

    def get_embedding(self,triplet_idx):
        return self._forward(triplet_idx)

    def forward(self,triplet_idx):
        return self.get_score(triplet_idx)

    def _forward(self, triplet_idx):
        head_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)

        # head_embedding = F.normalize(head_embedding, p=2, dim=2)
        # tail_embedding = F.normalize(tail_embedding, p=2, dim=2)
        relation_embedding = F.normalize(relation_embedding,p=2,dim=2)

        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding

        return output # (batch,3,embedding_dim)
        
