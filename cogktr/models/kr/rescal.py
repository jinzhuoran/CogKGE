import torch
import torch.nn as nn
import torch.nn.functional as F

class Rescal(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim):
        super(Rescal, self).__init__()
        self.embedding_dim = embedding_dim
        self.name = "Rescal"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.entity_embedding = nn.Embedding(entity_dict_len, embedding_dim)
        self.relation_embedding = nn.Embedding(relation_dict_len, embedding_dim * embedding_dim)
 

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def forward(self,sample):
        batch_h,batch_r,batch_t =  sample[:, 0], sample[:, 1], sample[:, 2]
        A = self.entity_embedding(batch_h).view(-1,1,self.embedding_dim) # (batch,1,embedding)
        R = self.relation_embedding(batch_r).view(-1,self.embedding_dim,self.embedding_dim) # (batch,embedding,embedding)
        A_T = self.entity_embedding(batch_t).view(-1,self.embedding_dim,1) # (batch,embedding,1)

        return torch.squeeze(torch.matmul(torch.matmul(A,R),A_T)) # (batch,)
    def get_score(self,sample):
        return self.forward(sample)
     