from numpy import positive
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    def __init__(self, margin,C=0):
        self.margin = margin
        self.C = C
        pass

    def __call__(self, positive_score, negative_score,penalty=0.0):
        output = torch.mean(F.relu(self.margin + positive_score - negative_score)) + self.C * penalty
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
    pass
        
class RotatELoss:
    pass

class TransALoss:
    pass

class KEPLERLoss:
    def __init__(self, margin):
        self.margin = margin

    def KELoss(self,positive_score,negative_score):
        positive_loss=(-1)*torch.log(torch.sigmoid(self.margin-positive_score)).type(torch.FloatTensor)
        negative_loss=(-1)*torch.log(torch.sigmoid(self.margin-negative_score)).type(torch.FloatTensor)
        keloss=torch.mean(positive_loss+negative_loss)

        return keloss

    # def MLMLoss(self):
    #     return 0.0

    def __call__(self, positive_score, negative_score):
        output_mean=self.KELoss(positive_score,negative_score)
        # output_mean=self.KELoss(positive_score,negative_score)+self.MLMLoss()
        return output_mean




        

