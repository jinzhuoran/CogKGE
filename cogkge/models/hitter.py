# import math
# import torch
# import torch.nn as nn
# from cogkge.models.basemodel import BaseModel
# class HittER(BaseModel):
#     def __init__(self,
#                  entity_dict_len=10,
#                  relation_dict_len=10,
#                  embedding_dim=320,
#                  dropout=0.1):
#         super().__init__(model_name="Hitter")
#         self.entity_dict_len=entity_dict_len
#         self.relation_dict_len=relation_dict_len
#         self.embedding_dim=embedding_dim
#         self.dropout=dropout
#
#         self.source_r_encoder=Entity_Transformer(entity_dict_len=entity_dict_len,
#                                                   relation_dict_len=relation_dict_len,
#                                                   embedding_dim=embedding_dim,
#                                                   dropout=dropout)
#
#         self.gcls_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
#         self.inter_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
#         self.other_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
#         self.transform=torch.nn.Linear(self.embedding_dim,self.entity_dict_len)
#         self.encoder_layer = torch.nn.TransformerEncoderLayer(
#             d_model=self.embedding_dim,
#             nhead=8,
#             dim_feedforward=128,
#             dropout=self.dropout,
#             activation='relu',
#         )
#         self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=6)
#
#     def _reset_param(self):
#         # 重置参数
#         nn.init.uniform_(self.gcls_emb.data)
#         nn.init.uniform_(self.inter_type_emb.data)
#         nn.init.uniform_(self.other_type_emb.data)
#
#     def forward(self, data):
#         # 前向传播
#         source=data[0].unsqueeze(1)
#         r=data[1].unsqueeze(1)
#         input=torch.cat((source,r),dim=1)
#         embeddings=self.source_r_encoder(input).permute(1,0,2)
#         embeddings=self.encoder_layer(embeddings)
#         distribution=self.transform(embeddings)[0]
#         return torch.sigmoid(distribution)
#
#
#
#     def loss(self, data):
#         # 计算损失
#         data= self.data_to_device(data)
#         source_r=data[:2]
#         label=data[-1]
#         distribution=self.forward(source_r)
#         return self.model_loss(distribution, label)
#
#
#     def data_to_device(self, data):
#         for index, item in enumerate(data):
#             data[index] = item.to(self.model_device)
#         return data
#
#     def metric(self, data):
#         # 模型评价
#         pass
#
#
# class Entity_Transformer(nn.Module):
#     def __init__(self,entity_dict_len=10,relation_dict_len=10,embedding_dim=320,dropout=0.1):
#         super().__init__()
#         self.entity_dict_len=entity_dict_len
#         self.relation_dict_len=relation_dict_len
#         self.embedding_dim=embedding_dim
#         self.dropout=dropout
#
#         self.e_embedding = nn.Embedding(self.entity_dict_len, self.embedding_dim)
#         self.r_embedding = nn.Embedding(self.relation_dict_len, self.embedding_dim)
#         self.cls_emb = nn.parameter.Parameter(torch.zeros(self.embedding_dim))
#         self.sub_type_emb = nn.parameter.Parameter(torch.zeros(self.embedding_dim))
#         self.rel_type_emb = nn.parameter.Parameter(torch.zeros(self.embedding_dim))
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embedding_dim,
#             nhead=8,
#             dim_feedforward=128,
#             dropout=self.dropout,
#             activation='relu',
#         )
#         self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=3)
#
#         self._reset_param()
#
#     def _reset_param(self):
#         nn.init.xavier_uniform_(self.e_embedding.weight.data)
#         nn.init.xavier_uniform_(self.r_embedding.weight.data)
#         nn.init.uniform_(self.cls_emb.data)
#         nn.init.uniform_(self.sub_type_emb.data)
#         nn.init.uniform_(self.rel_type_emb.data)
#         for layer in self.encoder.layers:
#             nn.init.xavier_uniform_(layer.linear1.weight.data)
#             nn.init.xavier_uniform_(layer.linear2.weight.data)
#             nn.init.xavier_uniform_(layer.self_attn.out_proj.weight.data)
#             if layer.self_attn._qkv_same_embed_dim:
#                 nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
#             else:
#                 nn.init.xavier_uniform_(layer.self_attn.q_proj_weight)
#                 nn.init.xavier_uniform_(layer.self_attn.k_proj_weight)
#                 nn.init.xavier_uniform_(layer.self_attn.v_proj_weight)
#
#     def forward(self,data):
#         h_embedding = self.e_embedding(data[:,0])
#         r_embedding = self.r_embedding(data[:,1])
#         batch_size = data.size()[0]
#         embeddings = torch.stack(
#             (
#                 self.cls_emb.repeat((batch_size, 1)),
#                 h_embedding + self.sub_type_emb.unsqueeze(0),
#                 r_embedding + self.rel_type_emb.unsqueeze(0),
#             ),
#             dim=0
#         )
#         embeddings = self.encoder(embeddings)
#         embeddings = embeddings[0, :].unsqueeze(1)
#         return embeddings

import torch
import torch.nn as nn


# TODO: add classification trainer
class HittER(torch.nn.Module):
    def __init__(self, embedding_dim=320, dropout=0.1):
        super(HittER, self).__init__()
        self.name = "HittER"
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.gcls_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.gcls_emb.data)
        self.inter_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.inter_type_emb.data)
        self.other_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.embedding_dim))
        nn.init.uniform_(self.other_type_emb.data)

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
        score = torch.nn.functional.softmax(out, dim=1)
        return score

    def forward(self, embeddings):
        return self.get_score(embeddings)


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
        t = self.entity_embedding(triplet_idx[:, 2])
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
    et = Entity_Transformer(entity_dict_len=100, relation_dict_len=10, embedding_dim=320, dropout=0.1)
    a = torch.IntTensor([[1, 2, 3], [3, 5, 6], [66, 7, 86]])
    b = et.get_embedding(a)
    c = torch.stack((b, b))
    hitter = HittER()
    hitter.get_score(c)
    print(1)