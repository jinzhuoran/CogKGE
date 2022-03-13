

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ..adapter import *
# from transformers import RobertaModel
#
#
# class TransE_Adapter(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50,node_lut=None,p=1,time_lut=None,time_embedding_dim=10):
#         super(TransE_Adapter, self).__init__()
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.embedding_dim = embedding_dim
#         self.node_lut=node_lut
#         self.time_lut=time_lut
#         self.p = p
#         self.name = "TransE"
#         self.time_embedding_dim=time_embedding_dim
#
#         self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
#         self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#
#         self.nodetype_transfer_matrix=None
#         if node_lut is not None and node_lut.type is not None:
#             self.nodetype_len=max(node_lut.type)+1
#             self.nodetype_transfer_matrix=nn.Embedding(num_embeddings=self.nodetype_len, embedding_dim=self.embedding_dim*self.embedding_dim)
#
#         self.pre_training_model=None
#         if node_lut is not None and node_lut.token is not None:
#             self.pre_training_model_name = "roberta-base"
#             self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)
#             self.out_dim=self.pre_training_model.pooler.dense.out_features
#             self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.out_dim)
#
#         self.time_transfer_matrix=None
#         if time_lut is not None:
#             self.time_transfer_matrix=nn.Embedding(num_embeddings=len(self.time_lut), embedding_dim=self.time_embedding_dim)
#
#     def get_score(self,triplet_idx):
#         output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
#         score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
#         return score  # (batch,)
#
#     def forward(self,triplet_idx):
#         return self.get_score(triplet_idx)
#
#     # @description
#     # @graph
#     @nodetype
#     # @time
#     def get_embedding(self,triplet_idx,
#                       node_lut=None,
#                       nodetype_transfer_matrix=None,
#                       embedding_dim=50,
#                       pre_training_model=None,
#                       time_transfer_matrix=None):
#         head_embedding = self.entity_embedding(triplet_idx[:, 0])
#         relation_embedding = self.relation_embedding(triplet_idx[:, 1])
#         tail_embedding = self.entity_embedding(triplet_idx[:, 2])
#         return head_embedding,relation_embedding,tail_embedding
#
#     def _forward(self, triplet_idx):
#         head_embedding,relation_embedding,tail_embedding=self.get_embedding(triplet_idx,
#                                                                             self.node_lut,
#                                                                             self.nodetype_transfer_matrix,
#                                                                             self.embedding_dim,
#                                                                             self.pre_training_model,
#                                                                             self.time_transfer_matrix)
#
#         head_embedding = F.normalize(head_embedding, p=2, dim=1)
#         tail_embedding = F.normalize(tail_embedding, p=2, dim=1)
#
#         head_embedding = torch.unsqueeze(head_embedding , 1)
#         relation_embedding = torch.unsqueeze(relation_embedding , 1)
#         tail_embedding = torch.unsqueeze(tail_embedding, 1)
#         triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)
#
#         output = triplet_embedding
#
#         return output # (batch,3,embedding_dim)
#

import torch.nn as nn
import torch.nn.functional as F
import torch
from cogkge.models.basemodel import BaseModel
from ..adapter import *


class TransE_Adapter(BaseModel):
    def __init__(self,
                 entity_dict_len,
                 relation_dict_len,
                 embedding_dim=50,
                 p_norm=1,
                 penalty_weight=0.0):
        super().__init__(model_name="TransE", penalty_weight=penalty_weight)
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.p_norm = p_norm

        self.e_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.r_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)

        self._reset_param()


    def _reset_param(self):
        # 重置参数
        nn.init.xavier_uniform_(self.e_embedding.weight.data)
        nn.init.xavier_uniform_(self.r_embedding.weight.data)

    def forward(self, batch):
        # 前向传播
        h, r, t = self.get_batch(batch)
        h_embedding, r_embedding, t_embedding = self.get_triplet_embedding(h=h, r=r, t=t,batch=batch)
        h_embedding = F.normalize(h_embedding, p=2, dim=1)
        t_embedding = F.normalize(t_embedding, p=2, dim=1)

        score = F.pairwise_distance(h_embedding + r_embedding, t_embedding, p=self.p_norm)

        return score

    def get_realation_embedding(self, relation_ids):
        # 得到关系的embedding
        return self.r_embedding(relation_ids)

    def get_entity_embedding(self, entity_ids):
        # 得到实体的embedding
        return self.e_embedding(entity_ids)

    @description
    # @graph
    # @nodetype
    # @time
    def get_triplet_embedding(self, h, r, t,batch):
        # 得到三元组的embedding
        h_embedding = self.e_embedding(h)
        r_embedding = self.r_embedding(r)
        t_embedding = self.e_embedding(t)
        return h_embedding, r_embedding, t_embedding

    def get_batch(self, data):
        # 得到一个batch的数据
        h = data["h"].to(self.model_device)
        r = data["r"].to(self.model_device)
        t = data["t"].to(self.model_device)
        return h, r, t

    def penalty(self):
        # 正则项
        penalty_loss = torch.tensor(0.0).to(self.model_device)
        for param in self.parameters():
            penalty_loss += torch.sum(param ** 2)
        return self.penalty_weight * penalty_loss

    def loss(self, batch):
        # 计算损失
        # h, r, t = self.get_batch(data)
        # pos_data = torch.cat((h.unsqueeze(1), r.unsqueeze(1), t.unsqueeze(1)), dim=1)
        # neg_data = self.model_negative_sampler.create_negative(pos_data)
        pos_score = self.forward(batch)
        # neg_score = self.forward(neg_data)
        return self.model_loss(pos_score, pos_score) + self.penalty()

    def metric(self, data):
        # 模型评价
        self.model_metric(data)


if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from cogkge.core.loss import MarginLoss
    from cogkge.core.sampler import UnifNegativeSampler
    from cogkge.core.metric import Link_Prediction
    from torch.utils.data import DataLoader, Dataset


    # trainer外部
    class MyDataset(Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def __len__(self):
            return data.shape[0]

        def __getitem__(self, index):
            return {"h": self.data[index][0],
                    "r": self.data[index][1],
                    "t": self.data[index][2]}


    data = torch.ones((100, 3), dtype=torch.long)
    model = TransE(entity_dict_len=3,
                   relation_dict_len=2,
                   embedding_dim=50,
                   p_norm=1)
    loss = MarginLoss(margin=1.0, reverse=False)
    metric = Link_Prediction(link_prediction_raw=True,
                             link_prediction_filt=False,
                             batch_size=5000000,
                             reverse=False)
    negative_sampler = UnifNegativeSampler(triples=data,
                                           entity_dict_len=3,
                                           relation_dict_len=2,
                                           device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_dataset = MyDataset(data)

    # trainer内部
    model.set_model_config(model_loss=loss,
                           model_metric=metric,
                           model_negative_sampler=negative_sampler,
                           model_device="cpu")
    train_loader = DataLoader(dataset=train_dataset, batch_size=10)
    train_epoch_loss = 0.0
    for batch in tqdm(train_loader):
        train_loss = model.loss(batch)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
    print("end")
