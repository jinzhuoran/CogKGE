import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(PyGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout()

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_feature
        self.out_features = out_feature
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
