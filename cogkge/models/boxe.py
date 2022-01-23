import torch
import torch.nn as nn


class BoxE(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim):
        """
        BoxE model for binary facts
        """
        super(BoxE, self).__init__()
        self.name = "BoxE"
        self.embedding_dim = embedding_dim
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len

        self.entity_embedding_base = nn.Embedding(entity_dict_len, embedding_dim)
        self.entity_embedding_trans = nn.Embedding(entity_dict_len, embedding_dim)
        self.relation_embedding_center1 = nn.Embedding(relation_dict_len, embedding_dim)
        self.relation_embedding_width1 = nn.Embedding(relation_dict_len, embedding_dim)
        self.relation_embedding_center2 = nn.Embedding(relation_dict_len, embedding_dim)
        self.relation_embedding_width2 = nn.Embedding(relation_dict_len, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embedding_base.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding_trans.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_center1.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_width1.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_center2.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_width2.weight.data)

    def forward(self, sample):
        batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]

        # (batch,embedding) every entity is represented by two vectors
        h_base = self.entity_embedding_base(batch_h)
        t_base = self.entity_embedding_base(batch_t)
        h_trans = self.entity_embedding_trans(batch_h)
        t_trans = self.entity_embedding_trans(batch_t)

        # (batch,embedding) embedding point for the head and tail
        h = h_base + t_trans
        t = t_base + h_trans

        # (batch,embedding) relation box r1(c1,w1) and r2(c2,w2)
        c1 = self.relation_embedding_center1(batch_r)
        c2 = self.relation_embedding_center2(batch_r)
        w1 = torch.abs(self.relation_embedding_width1(batch_r)) + 1
        w2 = torch.abs(self.relation_embedding_width2(batch_r)) + 1

        score = torch.norm(self.distance(h, c1, w1), p=2.0, dim=-1) + \
                torch.norm(self.distance(t, c2, w2), p=2.0, dim=-1)
        return score  # (batch,)

    def distance(self, e, c, w):
        """
        distance function for the given entity embedding(e) relative to a given target box(c,w)
        defined piece-wise over two cases
        e: (batch,embeddign_dim) entity embedding
        c: (batch,embedding_dim) relation box center
        w: (batch,embedding_dim) relation box width
        return: (batch,embedding_dim)
        """
        score = None
        K = 0.5 * (w - 1) * (w - torch.reciprocal(w))  # width_dependent factor to preserve function continuity
        if self._is_in_box(e, c, w):  # if the entity is in the target box
            score = torch.abs(e - c) / w  # element-wise division
        else:  # if the entity is not in the target box
            score = torch.abs(e - c) * w - K  # element-wise multiplication

        return score

    def _is_in_box(self, e, c, w):
        """
        determine if one embedding point is in the given relation box
        e: (batch,embeddign_dim) entity embedding
        c: (batch,embedding_dim) relation box center
        w: (batch,embedding_dim) relation box width
        return: True or False
        """
        u = c + (w - 1) / 2
        l = c - (w - 1) / 2

        if (torch.any(e > u).item() or torch.any(e < l).item()):
            return False
        else:
            return True
