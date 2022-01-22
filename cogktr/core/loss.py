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
    def __init__(self, margin, C=0):
        self.margin = margin
        self.C = C

    def __call__(self, positive_score, negative_score, penalty=0.0):
        output = torch.mean(F.relu(self.margin + positive_score - negative_score)) + self.C * penalty
        return output


class NegLogLikehoodLoss:
    def __init__(self, C=0):
        # self.lamda = lamda
        self.C = C

    def __call__(self, positive_score, negative_score, penalty=0):
        """
        positive_score: (batch,)
        negative_score: (batch,)
        """
        softplus = lambda x: torch.log(1 + torch.exp(x))
        output = softplus(- positive_score) + softplus(negative_score)  # (batch,)
        return torch.mean(output) + self.C * penalty


class NegSamplingLoss:
    def __init__(self, margin, alpha, neg_per_pos, C=0):
        self.alpha = alpha
        self.margin = margin
        self.neg_per_pos = neg_per_pos
        self.C = C

    def __call__(self, p_score, n_score, penalty):
        """
        p_score: (batch,)
        n_score: (batch * neg_per_pos,)
        return: tensor form scalar
        """
        # n_score = torch.cat(
        #     [torch.unsqueeze(n_score[i * self.neg_per_pos:(i + 1) * self.neg_per_pos], dim=1)
        #      for i in range(int(n_score.shape[0] / self.neg_per_pos))]
        #     , dim=-1
        # )  # tensor(neg_per_pos,batch)
        # n_score = n_score.transpose(0, 1)  # tensor(batch,neg_per_pos)

        n_score = n_score.reshape(self.neg_per_pos,-1).T  #add
        n_log_score = torch.log(torch.sigmoid(n_score - self.margin))
        n_prob = torch.exp(self.alpha * n_score) / torch.sum(torch.exp(self.alpha * n_score), dim=-1,
                                                             keepdim=True)  # (batch,neg_per_pos)

        negative_loss = -torch.sum(n_log_score * n_prob, dim=-1)  # (batch,)
        positive_loss = -torch.log(torch.sigmoid(self.margin - p_score))  # (batch,)
        return torch.mean(positive_loss + negative_loss) + self.C * penalty


# Still working on this...
class TransALoss:
    pass


class RotatELoss(nn.Module):
    def __init__(self): 
        super(RotatELoss,self).__init__()
     
    def forward(self, p_score, n_score,penalty=None):
        return torch.mean(-F.logsigmoid(p_score) - F.logsigmoid(-n_score))


class TuckERLoss:
    def __init__(self, margin):
        pass

    def __call__(self, p_score, n_score, penalty=None):
        p_score = -torch.mean(torch.log(p_score))
        n_score = -torch.mean(torch.log(1 - n_score))
        return (p_score + n_score) / 2


class KEPLERLoss:
    def __init__(self, margin):
        self.margin = margin

    def KELoss(self, positive_score, negative_score):
        positive_loss = (-1) * torch.log(torch.sigmoid(self.margin - positive_score)).type(torch.FloatTensor)
        negative_loss = (-1) * torch.log(torch.sigmoid(negative_score - self.margin)).type(torch.FloatTensor)
        keloss = torch.mean(positive_loss + negative_loss)

        return keloss

    # def MLMLoss(self):
    #     return 0.0

    def __call__(self, positive_score, negative_score):
        output_mean = self.KELoss(positive_score, negative_score)
        # output_mean=self.KELoss(positive_score,negative_score)+self.MLMLoss()
        return output_mean






