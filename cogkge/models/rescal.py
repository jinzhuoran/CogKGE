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

    def forward(self, sample):
        batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
        # A = self.entity_embedding(batch_h).view(-1,1,self.embedding_dim) # (batch,1,embedding)
        A = self.entity_embedding(batch_h)  # (batch,embedding)
        A = F.normalize(A, p=2, dim=-1)
        R = self.relation_embedding(batch_r).view(-1, self.embedding_dim,
                                                  self.embedding_dim)  # (batch,embedding,embedding)
        A_T = self.entity_embedding(batch_t).view(-1, self.embedding_dim, 1)  # (batch,embedding,1)
        A_T = F.normalize(A_T, p=2, dim=1)

        tr = torch.matmul(R, A_T)  # (batch,embedding_dim,1)
        tr = tr.view(-1, self.embedding_dim)  # (batch,embedding_dim)

        return -torch.sum(A * tr, dim=-1)  # (batch,)

        # return torch.squeeze(torch.matmul(torch.matmul(A,R),A_T)) # (batch,)

    def get_score(self, sample):
        return self.forward(sample)

    # def get_score(self,sample):
    #     output = self._forward(sample)
    #     score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
    #     return score  # (batch,) 

    # def get_embedding(self,sample):
    #     return self._forward(sample)

    # def forward(self,sample):
    #     return self.get_score(sample)

    # def _forward(self, sample):  # sample:(batch,3)
    #     batch_h,batch_r,batch_t =  sample[:, 0], sample[:, 1], sample[:, 2]
    #     h = self.entity_embedding(batch_h)
    #     r = self.relation_embedding(batch_r)
    #     t = self.entity_embedding(batch_t)

    #     r_norm = self.norm_vector(batch_r)

    #     h = self.transfer(h,r_norm)
    #     t = self.transfer(t,r_norm)

    #     h = F.normalize(h, p=2.0,dim=-1)
    #     r = F.normalize(r, p=2.0, dim=-1)
    #     t = F.normalize(t, p=2.0, dim=-1)

    #     h = torch.unsqueeze(h,1)
    #     r = torch.unsqueeze(r,1)
    #     t = torch.unsqueeze(t,1)

    #     return torch.cat((h,r,t),dim=1) # return:(batch,3,embedding_dim)
