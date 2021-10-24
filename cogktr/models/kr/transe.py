import torch
import torch.nn as nn
import torch.nn.functional as F
class TransE(nn.Module):
    def __init__(self,entity_dict_len,relation_dict_len,embedding_dim,margin,L):
        super(TransE, self).__init__()
        self.entity_dict_len=entity_dict_len
        self.relation_dict_len=relation_dict_len
        self.embedding_dim=embedding_dim
        self.margin=margin
        self.L=L
        self.name="TransE"
        self.square=embedding_dim**0.5
        self.entity_embedding=nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(-6/self.square, 6/self.square)
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=self.L, dim=1)
        self.relation_embedding=nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(-6/self.square, 6/self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=self.L, dim=1)
        self.distance = torch.nn.PairwiseDistance(self.L)

    def forward(self, positive_item):
        positive_item_head=torch.unsqueeze(self.entity_embedding(positive_item[:,0]), 1)
        positive_item_relation=torch.unsqueeze(self.relation_embedding(positive_item[:,1]), 1)
        positive_item_tail=torch.unsqueeze(self.entity_embedding(positive_item[:,2]), 1)

        positive_item_head=torch.unsqueeze(positive_item_head, 2)
        positive_item_relation=torch.unsqueeze(positive_item_relation, 2)
        positive_item_tail=torch.unsqueeze( positive_item_tail, 2)

        positive_embedding=torch.cat([positive_item_head,positive_item_relation,positive_item_tail],dim=1)

        output=positive_embedding

        return output