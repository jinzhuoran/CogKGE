import torch
import torch.nn as nn
import torch.nn.functional as F

class TuckER(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len,dim_entity,dim_relation):
        super(TuckER, self).__init__()
        self.name = "TuckER"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.dim_entity = dim_entity
        self.dim_relation = dim_relation
 
        self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
        self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
        self.core_tensor = nn.Parameter(
            torch.randn(dim_entity,dim_relation,dim_entity) 
        )

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.core_tensor.data)


    def forward(self,sample):
        batch_h,batch_r,batch_t =  sample[:, 0], sample[:, 1], sample[:, 2]
        # print(batch_h.shape)
        h = self.entity_embedding(batch_h) 
        r = self.relation_embedding(batch_r) # (batch,dim_relation)
        t = self.entity_embedding(batch_t)  # (batch,dim_entity)

        h = F.normalize(h, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)
        r = F.normalize(r, p=2.0, dim=-1)

        r = r.view(-1,1,1,self.dim_relation) # (batch,1,1,dim_relation)
        h = torch.unsqueeze(h,dim=-1) # (batch,dim_entity,1)
        t = torch.unsqueeze(t,dim=1)  # (batch,1,dim_entity)

        tmp = torch.matmul(r,self.core_tensor) # (batch,dim_entity,1,dim_entity)
        tmp = torch.squeeze(tmp)               # (batch,dim_entity,dim_etity)

        score = torch.bmm(t,tmp) # (batch,1,dim_entity)
        score = torch.bmm(score,h) # (batch,1,1)
        return torch.squeeze(score)  # (batch,)
 
    def get_score(self,sample):
        return self.forward(sample)
 