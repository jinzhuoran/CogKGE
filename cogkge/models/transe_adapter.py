import torch
import torch.nn as nn
import torch.nn.functional as F
from ..adapter import *
from transformers import RobertaModel


class TransE_Adapter(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50,node_lut=None,p=1,time_lut=None,time_embedding_dim=10):
        super(TransE_Adapter, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.node_lut=node_lut
        self.time_lut=time_lut
        self.p = p
        self.name = "TransE"
        self.time_embedding_dim=time_embedding_dim

        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

        self.nodetype_transfer_matrix=None
        if node_lut is not None and node_lut.type is not None:
            self.nodetype_len=max(node_lut.type)+1
            self.nodetype_transfer_matrix=nn.Embedding(num_embeddings=self.nodetype_len, embedding_dim=self.embedding_dim*self.embedding_dim)

        self.pre_training_model=None
        if node_lut is not None and node_lut.token is not None:
            self.pre_training_model_name = "roberta-base"
            self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)
            self.out_dim=self.pre_training_model.pooler.dense.out_features
            self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.out_dim)

        self.time_transfer_matrix=None
        if time_lut is not None:
            self.time_transfer_matrix=nn.Embedding(num_embeddings=len(self.time_lut), embedding_dim=self.time_embedding_dim)

    def get_score(self,triplet_idx):
        output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)

    def forward(self,triplet_idx):
        return self.get_score(triplet_idx)

    @description
    # @graph
    # @nodetype
    # @time
    def get_embedding(self,triplet_idx,
                      node_lut=None,
                      nodetype_transfer_matrix=None,
                      embedding_dim=50,
                      pre_training_model=None,
                      time_transfer_matrix=None):
        head_embedding = self.entity_embedding(triplet_idx[:, 0])
        relation_embedding = self.relation_embedding(triplet_idx[:, 1])
        tail_embedding = self.entity_embedding(triplet_idx[:, 2])
        return head_embedding,relation_embedding,tail_embedding

    def _forward(self, triplet_idx):
        head_embedding,relation_embedding,tail_embedding=self.get_embedding(triplet_idx,
                                                                            self.node_lut,
                                                                            self.nodetype_transfer_matrix,
                                                                            self.embedding_dim,
                                                                            self.pre_training_model,
                                                                            self.time_transfer_matrix)

        head_embedding = F.normalize(head_embedding, p=2, dim=1)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=1)

        head_embedding = torch.unsqueeze(head_embedding , 1)
        relation_embedding = torch.unsqueeze(relation_embedding , 1)
        tail_embedding = torch.unsqueeze(tail_embedding, 1)
        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding

        return output # (batch,3,embedding_dim)

