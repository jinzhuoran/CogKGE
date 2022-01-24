import torch
import torch.nn as nn


# [CLS] embedding
# source entity token embedding + source entity type embedding
# predicate token embedding + predicate type embedding
# dot product with [CLS] embedding and target entity embedding to produce score
class HittER(torch.nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=320, dropout=0.1):
        super(HittER, self).__init__()
        self.name = "HittER"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.entity_embedding = nn.Embedding(self.entity_dict_len, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.relation_dict_len, self.embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.cls_emb.data)
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.sub_type_emb.data)
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.rel_type_emb.data)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            dim_feedforward=1280,
            dropout=self.dropout,
            activation='relu',
        )
        self.encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=3
        )

        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.linear1.weight.data)
            nn.init.xavier_uniform_(layer.linear2.weight.data)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight.data)
            if layer.self_attn._qkv_same_embed_dim:
                nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            else:
                nn.init.xavier_uniform_(layer.self_attn.q_proj_weight)
                nn.init.xavier_uniform_(layer.self_attn.k_proj_weight)
                nn.init.xavier_uniform_(layer.self_attn.v_proj_weight)

    def get_score(self, triplet_idx):
        h, r, t = self.get_embedding(triplet_idx)
        batch_size = h.size()[0]
        embeddings = torch.stack(
            (
                self.cls_emb.repeat((batch_size, 1)),
                h + self.sub_type_emb.unsqueeze(0),
                r + self.rel_type_emb.unsqueeze(0),
            ),
            dim=0,
        )

        out = self.encoder.forward(embeddings)
        out = out[0, ::]
        score = torch.nn.functional.softmax(torch.mm(out, self.entity_embedding.weight.data.t()), dim=1)
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

    def loss(self, triplet_idx, loss_function=nn.CrossEntropyLoss()):
        score = self.forward(triplet_idx)
        target = torch.zeros(score.size(), dtype=torch.float)
        target = target.scatter_(1, triplet_idx[:, 2].view(score.size()[0], -1).to(torch.int64), 1)
        loss = loss_function(score, target)
        return loss


if __name__ == '__main__':
    hitter = HittER(entity_dict_len=100, relation_dict_len=10, embedding_dim=320, dropout=0.1)
    a = torch.IntTensor([[1, 2, 3], [3, 5, 6], [66, 7, 86]])
    score = hitter.loss(a)
    print(1)
