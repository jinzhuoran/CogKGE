import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(torch.nn.Module):
    def __init__(self, margin, reverse=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        if not isinstance(reverse, bool):
            raise TypeError("param reverse must be True or False!")
        self.reverse = 1 if reverse == False else -1

    def forward(self, positive_score, negative_score, penalty=0.0):
        output = torch.mean(F.relu(self.margin + self.reverse * (positive_score - negative_score)))
        return output


class NegLogLikehoodLoss(torch.nn.Module):
    def __init__(self):
        super(NegLogLikehoodLoss, self).__init__()

    def forward(self, positive_score, negative_score):
        """
        positive_score: (batch,)
        negative_score: (batch,)
        """
        softplus = lambda x: torch.log(1 + torch.exp(x))
        output = softplus(- positive_score) + softplus(negative_score)  # (batch,)
        return torch.mean(output)


class NegSamplingLoss(torch.nn.Module):
    def __init__(self, alpha, neg_per_pos, C=0):
        super(NegSamplingLoss, self).__init__()
        self.alpha = alpha
        self.neg_per_pos = neg_per_pos

    def forward(self, p_score, n_score):
        """
        p_score: (batch,)
        n_score: (batch * neg_per_pos,)
        return: tensor form scalar
        """

        n_score = n_score.reshape(-1, self.neg_per_pos)  # (batch,neg_per_pos)
        negative_loss = (F.softmax(n_score * self.alpha, dim=1).detach() * F.logsigmoid(-n_score)).sum(
            dim=1)  # (batch,)
        positive_loss = F.logsigmoid(p_score)
        return torch.mean(-positive_loss - negative_loss)
