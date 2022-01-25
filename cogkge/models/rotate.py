import torch
import torch.nn as nn


class RotatE(torch.nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim, mode='head-batch', gamma=6.0, epsilon=2.0):
        super(RotatE, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.entity_dim = embedding_dim * 2
        self.relation_dim = embedding_dim
        self.name = "RotatE"
        self.mode = mode
        self.gamma = gamma
        self.epsilon = epsilon
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / embedding_dim]),
            requires_grad=False
        )
        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.entity_dim)
        nn.init.uniform_(
            tensor=self.entity_embedding.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def get_score(self, triplet_idx):
        head_embedding, relation_embedding, tail_embedding = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head_embedding, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail_embedding, 2, dim=1)
        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation_embedding / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if self.mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=1)
        return score


    def get_embedding(self, triplet_idx):
        return self._forward(triplet_idx)

    def forward(self, triplet_idx):
        return self.get_score(triplet_idx)

    def _forward(self, triplet_idx):
        head_embedding = self.entity_embedding(triplet_idx[:, 0])
        relation_embedding = self.relation_embedding(triplet_idx[:, 1])
        tail_embedding = self.entity_embedding(triplet_idx[:, 2])
        return head_embedding, relation_embedding, tail_embedding
