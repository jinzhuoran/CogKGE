import torch 
import torch.nn as nn
import torch.nn.functional as F

class TransA(nn.Module):
    def __init__(self,entity_dict_len,relation_dict_len,embedding_dim):
        super(TransA,self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.entity_embedding = nn.Embedding(entity_dict_len, embedding_dim)
        self.relation_embedding = nn.Embedding(relation_dict_len, embedding_dim)
        self.norm_vector = nn.Embedding(relation_dict_len,embedding_dim)
        self.relation_loss_embedding = nn.Embedding(self.relation_dict_len,embedding_dim * embedding_dim)
        self.Wr = None
        self.embedding_dim = embedding_dim
        self.name = "TransA"

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def get_penalty(self):
        return torch.norm(self.Wr)

    def get_score(self,sample):
        sample = self._forward(sample)
        h,r,t = sample[:,0],sample[:,1],sample[:,2]
        # Wr = self.relation_loss_embedding(r).view(-1,self.embedding_dim,self.embedding_dim)

        tmp_first = torch.unsqueeze(torch.abs(h+r-t),dim=1) # (batch,1,embedding_size)
        tmp_last = torch.unsqueeze(torch.abs(h+r-t),dim=-1) # (batch,embedding_size,1)
        return torch.squeeze(torch.bmm(torch.bmm(tmp_first,self.Wr),tmp_last))  # (batch,1,1) -> (batch,)

    def forward(self,sample):
        return self.get_score(sample)    
    
    def get_embedding(self,sample):
        return self._forward(sample)
        

    def _forward(self,sample): # sample:(batch,3)
        batch_h,batch_r,batch_t = sample[:,0],sample[:,1],sample[:,2]

        h = self.entity_embedding(batch_h)
        r = self.relation_embedding(batch_r)
        t = self.entity_embedding(batch_t)
        self.Wr = self.relation_loss_embedding(batch_r).view(-1,self.embedding_dim,self.embedding_dim)
        self.Wr = torch.abs(self.Wr)

        h = F.normalize(h, p=2.0,dim=-1)
        r = F.normalize(r, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)

        h = torch.unsqueeze(h,1)
        r = torch.unsqueeze(r,1)
        t = torch.unsqueeze(t,1)


        return torch.cat((h,r,t),dim=1) # return:(batch,3,embedding_dim)
 


       







