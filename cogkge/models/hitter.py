import math
import torch
import torch.nn as nn


class HittER(torch.nn.Module):
    def __init__(self, embedding_dim=320, dropout=0.1,entity_dict_len=100,relation_dict_len=10):
        super(HittER, self).__init__()
        self.name = "HittER"
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.entity_dict_len=entity_dict_len
        self.relation_dict_len=relation_dict_len

        self.gcls_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.gcls_emb.data)
        self.inter_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.inter_type_emb.data)
        self.other_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.other_type_emb.data)
        self.transform=torch.nn.Linear(embedding_dim,self.entity_dict_len)
        self.et = Entity_Transformer(entity_dict_len=entity_dict_len,
                                     relation_dict_len= relation_dict_len,
                                     embedding_dim=embedding_dim,
                                     dropout=dropout)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            dim_feedforward=1280,
            dropout=self.dropout,
            activation='relu',
        )
        self.encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=6
        )

    def get_embedding(self, embeddings):
        return self._forward(embeddings)

    def _forward(self, embeddings):
        batch_size = embeddings.size()[0]
        out = torch.concat(
            (
                self.gcls_emb.repeat((batch_size, 1)).unsqueeze(dim=1),
                (embeddings[:, 0, :] + self.inter_type_emb.unsqueeze(0)).unsqueeze(dim=1),
                embeddings[:, 1:, :] + self.other_type_emb.unsqueeze(0),
            ),
            dim=1,
        )
        out = self.encoder.forward(out)
        out = out[:, 0, :]
        return out

    def get_score(self, embeddings):
        out = self.get_embedding(embeddings)
        out=self.transform(out)
        score = torch.nn.functional.softmax(out, dim=1)
        return score

    def forward(self, embeddings):
        batch_size=embeddings.shape[0]
        et_embedding = self.et.get_embedding(embeddings)[:(int(math.floor(batch_size/3))*3)].reshape((int(math.floor(batch_size/3)), 3,self.embedding_dim))
        return self.get_score(et_embedding)

    def get_sampel_label_index(self,batch_size):
        return torch.arange(0, (batch_size//3)*3, step = 3)



# [CLS] embedding
# source entity token embedding + source entity type embedding
# predicate token embedding + predicate type embedding
# dot product with [CLS] embedding and target entity embedding to produce score
# Entity Transformer
class Entity_Transformer(torch.nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=320, dropout=0.1):
        super(Entity_Transformer, self).__init__()
        self.name = "Entity Transformer"
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
        embeddings = self.get_embedding(triplet_idx)
        score = torch.nn.functional.softmax(torch.mm(embeddings, self.entity_embedding.weight.data.t()), dim=1)
        return score

    def forward(self, triplet_idx):
        return self.get_score(triplet_idx)

    def get_embedding(self, triplet_idx):
        return self._forward(triplet_idx)

    def _forward(self, triplet_idx):
        h = self.entity_embedding(triplet_idx[:, 0])
        r = self.relation_embedding(triplet_idx[:, 1])
        # t = self.entity_embedding(triplet_idx[:, 2])
        batch_size = h.size()[0]
        embeddings = torch.stack(
            (
                self.cls_emb.repeat((batch_size, 1)),
                h + self.sub_type_emb.unsqueeze(0),
                r + self.rel_type_emb.unsqueeze(0),
            ),
            dim=0,
        )
        embeddings = self.encoder.forward(embeddings)
        embeddings = embeddings[0, ::]
        return embeddings

    def loss(self, triplet_idx, loss_function=nn.CrossEntropyLoss()):
        score = self.forward(triplet_idx)
        target = torch.zeros(score.size(), dtype=torch.float)
        target = target.scatter_(1, triplet_idx[:, 2].view(score.size()[0], -1).to(torch.int64), 1)
        loss = loss_function(score, target)
        return loss


if __name__ == '__main__':
    # et = Entity_Transformer(entity_dict_len=100, relation_dict_len=10, embedding_dim=320, dropout=0.1)
    hitter = HittER(embedding_dim=320, dropout=0.1, entity_dict_len=100,relation_dict_len=10)
    input=torch.IntTensor([[1, 2], [3, 5], [66, 7], [11, 2], [31, 6], [66, 4]])
    # input = torch.IntTensor([[1, 2, 3], [3, 5, 6], [66, 7, 86], [11, 2, 31], [31, 6, 61], [66, 4, 86]])
    # et_embedding = et.get_embedding(input).reshape((2,3,320))
    output=hitter(input )
    id=hitter.get_sampel_label_index(5)
    print(1)
