import torch
import torch.nn as nn
import torch.nn.functional as F


class TransD(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, dim_entity,dim_relation,negative_sample_method):
        super(TransD, self).__init__()
        self.name = "TransD"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.negative_sample_method = negative_sample_method
        self.dim_entity = dim_entity
        self.dim_relation = dim_relation

        self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
        self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
        self.entity_transfer = nn.Embedding(entity_dict_len,dim_entity)
        self.relation_transfer = nn.Embedding(relation_dict_len,dim_relation)

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.entity_transfer.weight.data)
        nn.init.xavier_uniform_(self.relation_transfer.weight.data)

    def transfer(self,e,e_transfer,r_transfer):
        return F.normalize(torch.sum(e * e_transfer,-1,True) * r_transfer + self.resize(e),
                          p=2,
                          dim=-1)


    def resize(self,e):
        if self.dim_entity == self.dim_relation:
            return e
        if self.dim_entity > self.dim_relation:
            return torch.narrow(e,-1,0,self.dim_relation)
        paddings = [0,0,0,self.dim_relation - self.dim_entity]
        return F.pad(e,paddings=paddings,mode="constant",value=0)

    def get_score(self,sample):
        output = self._forward(sample)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
        return score  # (batch,) 
    
    def forward(self,sample):
        return self.get_score(sample)

    def get_embedding(self,sample):
        return self._forward(sample)

    def _forward(self, sample):  # sample:(batch,3)
        batch_h,batch_r,batch_t =  sample[:, 0], sample[:, 1], sample[:, 2]
        h = self.entity_embedding(batch_h)
        r = self.relation_embedding(batch_r)
        t = self.entity_embedding(batch_t)

        h_transfer = self.entity_transfer(batch_h)
        r_transfer = self.relation_transfer(batch_r)
        t_transfer = self.entity_transfer(batch_t)

        h = self.transfer(h,h_transfer,r_transfer)
        t = self.transfer(t,t_transfer,r_transfer)

        h = F.normalize(h, p=2.0,dim=-1)
        r = F.normalize(r, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)

        h = torch.unsqueeze(h,1)
        r = torch.unsqueeze(r,1)
        t = torch.unsqueeze(t,1)

        return torch.cat((h,r,t),dim=1)

 
