import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim, negative_sample_method):
        super(TransE, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.negative_sample_method = negative_sample_method
        self.name = "TransE"
        self.square = embedding_dim ** 0.5
        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)

    def forward(self, triplet_idx):
        head_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embeddiing = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)

        head_embeddiing = F.normalize(head_embeddiing, p=2, dim=2)
        tail_embeddiing = F.normalize(tail_embeddiing, p=2, dim=2)

        triplet_embedding = torch.cat([head_embeddiing, relation_embeddiing, tail_embeddiing], dim=1)

        output = triplet_embedding

        return output
