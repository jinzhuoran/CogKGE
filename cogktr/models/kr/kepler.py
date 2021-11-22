import torch
import torch.nn as nn
import torch.nn.functional as F


#还没写完
class KEPLER(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, token_length,embedding_dim):
        super(KEPLER, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.token_length=token_length
        self.name = "KEPLER"
        # self.square = embedding_dim ** 0.5
        # self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        # self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
        #     -6 / self.square, 6 / self.square)
        # self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
        # self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        # self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
        #     -6 / self.square, 6 / self.square)
        # self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)

    def get_score(self,triplet_idx):
        pass
        # output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        # score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
        # return score  # (batch,)

    def forward(self,triplet_idx):
        pass
        # return self.get_score(triplet_idx)


    def get_embedding(self,triplet_idx):
        pass
        # return self._forward(triplet_idx)


    def _forward(self, triplet_idx):
        pass
        # head_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        # relation_embeddiing = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        # tail_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)
        #
        # head_embeddiing = F.normalize(head_embeddiing, p=2, dim=2)
        # tail_embeddiing = F.normalize(tail_embeddiing, p=2, dim=2)
        #
        # triplet_embedding = torch.cat([head_embeddiing, relation_embeddiing, tail_embeddiing], dim=1)
        #
        # output = triplet_embedding
        #
        # return output # (batch,3,embedding_dim)

if __name__=="__main__":
    import torch
    import numpy as np
    from torch.utils.data import Dataset
    import torch.utils.data as Data
    data_numpy=np.random.randn(25,3,30)
    print(data_numpy.shape)
    class MyDataset(Dataset):
        def __init__(self,data_numpy):
            self.data_numpy=data_numpy
        def __getitem__(self,index):
            return self.data_numpy[index]
            # return torch.tensor(self.data_numpy[index])
        def __len__(self):
            return len(self.data_numpy[:,0])
    dataset=MyDataset(data_numpy)
    data_loader=Data.DataLoader(dataset=dataset,batch_size=6,shuffle=True)
    model=KEPLER(entity_dict_len=10, relation_dict_len=5, token_length=30,embedding_dim=768)
    for epoch in range(1):
        for step,data in enumerate(data_loader):
            # 输入格式（batchsize,3,token_len）
            output=model(data)

            print("epoch:",epoch,"step:",step,"data:",data.shape)

    print("end")



