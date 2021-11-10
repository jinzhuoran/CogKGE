from numpy import positive
import torch
import torch.nn.functional as F


class MarginLoss:
    def __init__(self, margin):
        self.margin = margin
        pass

    def __call__(self, positive_batch, negtive_batch):
        positive_distance = F.pairwise_distance(positive_batch[:, 0] + positive_batch[:, 1], positive_batch[:, 2], p=2)
        negtive_distance = F.pairwise_distance(negtive_batch[:, 0] + negtive_batch[:, 1], negtive_batch[:, 2], p=2)
        output = torch.sum(F.relu(self.margin + positive_distance - negtive_distance)) / (positive_batch.shape[0])
        return output
    
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

