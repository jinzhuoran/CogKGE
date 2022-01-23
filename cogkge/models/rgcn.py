import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCN(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim, gamma=6.0):
        super(RGCN, self).__init__()
        self.name = "RGCN"
        self.embedding_dim = embedding_dim
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len

        self.entity_embedding = nn.Embedding(entity_dict_len, embedding_dim)
        self.relation_embedding_head = nn.Embedding(relation_dict_len, embedding_dim)
        self.relation_embedding_tail = nn.Embedding(relation_dict_len, embedding_dim)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_head.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_head.weight.data)

    def forward(self, sample):
        batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]

        h = self.entity_embedding(batch_h)
        r_h = self.relation_embedding_head(batch_r)
        r_t = self.relation_embedding_tail(batch_r)
        t = self.entity_embedding(batch_t)  # (batch,dim_entity)

        # constraint is only added on entity embeddings
        h = F.normalize(h, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)

        score = torch.norm(h * r_h - t * r_t, p=1, dim=-1)  # (batch,)
        return self.gamma.item() - score

    def get_score(self, sample):
        return self.forward(sample)
