from numpy import positive
import torch
import torch.nn as nn
import torch.nn.functional as F


# class MarginLoss:
#     def __init__(self, margin):
#         self.margin = margin
#         pass

#     def __call__(self, positive_batch, negtive_batch):
#         positive_distance = F.pairwise_distance(positive_batch[:, 0] + positive_batch[:, 1], positive_batch[:, 2], p=2)
#         negtive_distance = F.pairwise_distance(negtive_batch[:, 0] + negtive_batch[:, 1], negtive_batch[:, 2], p=2)
#         output = torch.sum(F.relu(self.margin + positive_distance - negtive_distance)) / (positive_batch.shape[0])
#         return output

class MarginLoss:
    def __init__(self, margin):
        self.margin = margin
        pass

    def __call__(self, positive_score, negative_score):
        output = torch.mean(F.relu(self.margin + positive_score - negative_score))
        return output

class NegLogLikehoodLoss:
    def __init__(self,lamda):
        self.lamda = lamda
    
    def __call__(self,positive_score,negative_score):
        """
        positive_score: (batch,)
        negative_score: (batch,)
        """
        softplus = lambda x:torch.log(1+torch.exp(x))
        output = softplus(- positive_score) + softplus(negative_score) # (batch,)
        return torch.mean(output)

# Still working on this...
class TransALoss:
    def __init__(self,margin):
        self.margin = margin
    
    def __call(self,positive_batch,negative_batch):
        h,r,t = positive_batch[:,0],positive_batch[:,1],positive_batch[:,2]
        h_,r_,t_ = negative_batch[:,0],negative_batch[:,1],negative_batch[:,3]
        
class RotatELoss:
    def __init__(self, margin):
        self.margin = margin
        pass

    def __call__(self, positive_batch, negtive_batch):
        positive_distance = F.pairwise_distance(positive_batch[:, 0] * positive_batch[:, 1], positive_batch[:, 2], p=2)
        negtive_distance = F.pairwise_distance(negtive_batch[:, 0] * negtive_batch[:, 1], negtive_batch[:, 2], p=2)
        output = torch.sum(F.relu(self.margin + positive_distance - negtive_distance)) / (positive_batch.shape[0])
        # print(positive_batch[:,0])
        # print(output.item())
        return output

# Stil working on this TransALoss...
class TransALoss(nn.Module):
    def __init__(self,margin,relation_dict_len,embedding_dim):
        super(TransALoss,self).__init__()
        self.margin = margin
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.relation_embedding = nn.Embedding(self.relation_dict_len,embedding_dim * embedding_dim)

    def get_score(self,h,r,t,Wr):
        """
        h,r,t:(batch,embedding_size)
        Wr:(batch,embedding_size,embedding_size)
        """
        tmp_first = torch.unsqueeze(torch.abs(h+r-t),dim=1) # (batch,1,embedding_size)
        tmp_last = torch.unsqueeze(torch.abs(h+r-t),dim=-1) # (batch,embedding_size,1)
        return torch.squeeze(torch.bmm(torch.bmm(tmp_first,Wr),tmp_last))  # (batch,1,1) -> (batch,)
        

    def forward(self,positive_batch,negative_batch):
        h,r,t = positive_batch[:,0],positive_batch[:,1],positive_batch[:,2] # (batch,embedding_size)
        h_,r_,t_ = negative_batch[:,0],negative_batch[:,1],negative_batch[:,2]
        Wr = self.relation_embedding(r).view(-1,self.embedding_size,self.embedding_size)
        positive_distance = self.get_score(h,r,t,Wr)
        negative_distance = self.get_score(h_,r_,t_,Wr)
        output = torch.mean(F.relu(self.margin + positive_distance - negative_distance))
        return output


        

