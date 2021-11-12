import torch
import torch.nn as nn
import torch.nn.functional as F

class TransH(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,negative_sample_method):
        super(TransH, self).__init__()
        self.name = "TransH"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.entity_embedding = nn.Embedding(entity_dict_len, embedding_dim)
        self.relation_embedding = nn.Embedding(relation_dict_len, embedding_dim)
        self.norm_vector = nn.Embedding(relation_dict_len,embedding_dim)
        self.negative_sample_method = negative_sample_method

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def transfer(self,e,norm):
        norm = F.normalize(norm,p=2.0,dim=-1)
        return e - torch.sum(e * norm,-1,True) * norm

    def get_score(self,sample):
        output = self.forward(sample)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
        return score  # (batch,) 

    def get_embedding(self,sample):
        return self.forward(sample)

    def forward(self, sample):  # sample:(batch,3)
        batch_h,batch_r,batch_t =  sample[:, 0], sample[:, 1], sample[:, 2]
        h = self.entity_embedding(batch_h)
        r = self.relation_embedding(batch_r)
        t = self.entity_embedding(batch_t)

        r_norm = self.norm_vector(batch_r)
        
        h = self.transfer(h,r_norm)
        t = self.transfer(t,r_norm)

        h = F.normalize(h, p=2.0,dim=-1)
        r = F.normalize(r, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)

        h = torch.unsqueeze(h,1)
        r = torch.unsqueeze(r,1)
        t = torch.unsqueeze(t,1)


        return torch.cat((h,r,t),dim=1) # return:(batch,3,embedding_dim)
 