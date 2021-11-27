import torch
import torch.nn as nn
import torch.nn.functional as F


class TransR(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, dim_entity,dim_relation,p=2.0):
        super(TransR, self).__init__()
        self.name = "TransR"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.dim_entity = dim_entity
        self.dim_relation = dim_relation
        self.p = p

        self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
        self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
        self.transfer_matrix = nn.Embedding(relation_dict_len,dim_entity * dim_relation)

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

    def transfer(self,e,r_transfer):
        r_transfer = r_transfer.view(-1,self.dim_entity,self.dim_relation)
        e = torch.unsqueeze(e,1)

        return torch.squeeze(torch.bmm(e,r_transfer))

    def get_score(self,sample):
        output = self._forward(sample)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,) 

    def get_embedding(self,sample):
        return self._forward(sample)

    def forward(self,sample):
        return self.get_score(sample)

    def _forward(self, sample):  # sample:(batch,3)
        batch_h,batch_r,batch_t =  sample[:, 0], sample[:, 1], sample[:, 2]
        h = self.entity_embedding(batch_h)
        r = self.relation_embedding(batch_r)
        t = self.entity_embedding(batch_t)

        r_transfer = self.transfer_matrix(batch_r)

        h = F.normalize(h, p=2.0,dim=-1)
        t = F.normalize(t, p=2.0, dim=-1) # ||h|| <= 1  ||t|| <= 1

        h = self.transfer(h,r_transfer)
        t = self.transfer(t,r_transfer)

        h = F.normalize(h, p=2.0,dim=-1)
        r = F.normalize(r, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)

        h = torch.unsqueeze(h,1)
        r = torch.unsqueeze(r,1)
        t = torch.unsqueeze(t,1)

        return torch.cat((h,r,t),dim=1)

 
