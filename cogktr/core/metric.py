# import torch
# import torch.utils.data as Data
# import torch.nn.functional as F
# import numpy as np
# import copy
# import math
# from tqdm import tqdm
#
#
# class Link_Prediction:
#     def __init__(self, entity_dict_len, batch_size):
#         self.entity_dict_len = entity_dict_len
#         self.batch_size = batch_size
#         self.name = "Link_Prediction"
#         self.total_rank = None
#         self.total_rank_numpy = None
#
#         self.MRR = []
#         self.MRR_numpy = None
#
#         self.raw_meanrank = None
#         self.raw_hitatten = None
#         self.raw_MRR = None
#
#     def __call__(self, model, metric_dataset, device):
#         metric_loader = Data.DataLoader(dataset=metric_dataset, batch_size=1, shuffle=False)
#         self.total_rank = list()
#         for step, batch_sample in enumerate(tqdm(metric_loader)):
#             # replace the tail
#             metric_single = copy.deepcopy(batch_sample)
#             x = metric_single[:, 2][0].numpy()
#             metric_single = metric_single.expand(self.entity_dict_len, 3)
#             metric_single = metric_single[:, :2]
#             new_tail = torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
#             metric_single = torch.cat([metric_single, new_tail], dim=1)
#             metric_single = metric_single.to(device)
#             # metric_distance = model.get_score(metric_single)
#             # metric_distance = model(metric_single)
#             metric_distance = []
#             for i in range(math.ceil(self.entity_dict_len / self.batch_size)):
#                 start = i * self.batch_size
#                 end = min((i + 1) * self.batch_size, self.entity_dict_len)
#                 data_input = metric_single[start:end, :]
#                 result = model(data_input).data.cpu().numpy()
#                 metric_distance.append(result)
#
#             # metric_distance = torch.cat([
#             #     model(metric_single[i * self.batch_size:min((i+1)*self.batch_size,self.entity_dict_len),:])
#             #         for i in range(math.ceil(self.entity_dict_len/self.batch_size))
#             # ])
#             # metric_total_matrix = np.argsort(metric_distance.data.cpu().numpy())
#             # metric_total_matrix = np.argsort(np.array(metric_distance).ravel('C'))
#             metric_total_matrix = np.argsort(np.concatenate([x for x in metric_distance]))
#             # metric_total_matrix = np.argsort(np.concatenate([x for x in metric_distance]))[::-1]
#             rank_tail = np.where(metric_total_matrix == x)[0][0]
#             self.total_rank.append(rank_tail)
#             self.MRR.append(1 / (rank_tail + 1))
#
#             # replace the head
#             metric_single = batch_sample
#             x = metric_single[:, 0][0].numpy()
#             metric_single = metric_single.expand(self.entity_dict_len, 3)
#             metric_single = metric_single[:, 1:]
#             new_head = torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
#             metric_single = torch.cat([new_head, metric_single], dim=1)
#             metric_single = metric_single.to(device)
#             # metric_distance = model.get_score(metric_single)
#             # metric_distance = model(metric_single)
#             # metric_distance = torch.cat([
#             #     model(metric_single[i * self.batch_size:min((i+1)*self.batch_size,self.entity_dict_len)])
#             #         for i in range(math.ceil(self.entity_dict_len/self.batch_size))
#             # ])
#             metric_distance = []
#             for i in range(math.ceil(self.entity_dict_len / self.batch_size)):
#                 start = i * self.batch_size
#                 end = min((i + 1) * self.batch_size, self.entity_dict_len)
#                 data_input = metric_single[start:end, :]
#                 result = model(data_input).data.cpu().numpy()
#                 metric_distance.append(list(result))
#
#             metric_total_matrix = np.argsort(np.concatenate([x for x in metric_distance]))
#             # metric_total_matrix = np.argsort(np.concatenate([x for x in metric_distance]))[::-1]
#             # metric_total_matrix = np.argsort(np.array(metric_distance).ravel('C'))
#             rank_head = np.where(metric_total_matrix == x)[0][0]
#             self.total_rank.append(rank_head)
#             self.MRR.append(1 / (rank_head + 1))
#
#         self.total_rank_numpy = np.array(self.total_rank)
#         self.MRR_numpy = np.array(self.MRR)
#         self.raw_meanrank = np.mean(self.total_rank_numpy)
#         self.raw_hitatten = np.sum(self.total_rank_numpy <= 9) / (2 * len(metric_dataset)) * 100
#         self.raw_MRR = np.mean(self.MRR_numpy)
#
#
# #

import torch
import torch.utils.data as Data
import numpy as np
import copy
import math
from tqdm import tqdm


class Link_Prediction:
    def __init__(self, entity_dict_len, batch_size,reverse=False):
        self.entity_dict_len = entity_dict_len
        self.batch_size = batch_size
        self.reverse=reverse
        self.name = "Link_Prediction"
        self.model = None
        self.device = None
        self.rank = list()
        self.rank_numpy = None
        self.MRR = list()
        self.MRR_numpy = None
        self.raw_meanrank = None
        self.raw_hitatten = None
        self.raw_MRR = None
        self.type_dict={"head":0,"tail":2}

    def _change_metric_node(self,single_sample,entity_type):
        single_triplet=copy.deepcopy(single_sample)
        entity_type_id=int(self.type_dict[entity_type])
        entity_id=single_triplet[:,entity_type_id][0].item()
        expand_triplet=single_triplet.expand(self.entity_dict_len, 3)
        if entity_type=="head":
            original_r_t = expand_triplet[:, 1:]
            expand_entity = torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
            expand_triplet = torch.cat([expand_entity, original_r_t], dim=1)
        if entity_type=="tail":
            original_h_r=expand_triplet[:, :2]
            expand_entity= torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
            expand_triplet= torch.cat([original_h_r,expand_entity], dim=1)
        expand_triplet = expand_triplet.to(self.device)
        distance = list()
        for i in range(math.ceil(self.entity_dict_len / self.batch_size)):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, self.entity_dict_len)
            data_input = expand_triplet[start:end, :]
            result = self.model(data_input).data.cpu().numpy()
            distance.append(result)
        total_distance= np.argsort(np.concatenate([x for x in distance]))
        single_rank= np.where(total_distance == entity_id)[0][0]+1
        single_rank=self.entity_dict_len+1-single_rank if self.reverse else single_rank
        self.rank.append(single_rank)
        self.MRR.append(1 / (single_rank))

    def __call__(self, model, metric_dataset, device):
        self.model=model
        self.device=device
        metric_loader = Data.DataLoader(dataset=metric_dataset, batch_size=1, shuffle=False)
        self.rank = list()
        self.MRR = list()
        for step, single_sample in enumerate(tqdm(metric_loader)):
            self._change_metric_node(single_sample,"tail")
            self._change_metric_node(single_sample, "head")
        self.rank_numpy = np.array(self.rank)
        self.MRR_numpy = np.array(self.MRR)
        self.raw_meanrank = np.mean(self.rank_numpy)
        # self.raw_hitatten = np.sum(self.rank_numpy <= 10) / ( len(metric_dataset)) * 100
        self.raw_hitatten = np.sum(self.rank_numpy <= 10) / (2 * len(metric_dataset)) * 100
        self.raw_MRR = np.mean(self.MRR_numpy)

if __name__=="__main__":
    metric=Link_Prediction(entity_dict_len=10, batch_size=3,reverse=False)
    metric_dataset=torch.tensor([[1,3,4],
                                [9,4,4],
                                [4,3,2],
                                [6,2,6],
                                [3,1,2],
                                [2,7,8],
                                [8,2,4],
                                [1,5,7]])
    def model(metric_dataset):
        score=metric_dataset[:,0]*0.1+metric_dataset[:,1]-metric_dataset[:,2]*0.3
        return score
    print(model(metric_dataset))
    metric(model=model, metric_dataset=metric_dataset, device="cuda:0")
    print(metric.raw_meanrank)
    print(metric.raw_hitatten)
    print(metric.raw_MRR)

# import torch
# import torch.utils.data as Data
# import torch.nn.functional as F
# import numpy as np
#
#
# class Link_Prediction:
#     def __init__(self, entity_dict_len):
#         self.entity_dict_len = entity_dict_len
#         self.name = "Link_Prediction"
#         self.total_rank = None
#         self.total_rank_numpy = None
#         self.raw_meanrank = None
#         self.raw_hitatten = None
#
#     def __call__(self, model, metric_dataset,device):
#         metric_loader = Data.DataLoader(dataset=metric_dataset, batch_size=1, shuffle=False)
#         self.total_rank = list()
#         for step, metric_single in enumerate(metric_loader):
#             x = metric_single[:, 2][0].numpy()
#             metric_single = metric_single.expand(self.entity_dict_len, 3)
#             metric_single = metric_single[:, :2]
#             new_tail = torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
#             metric_single = torch.hstack((metric_single, new_tail))
#             metric_single = metric_single.to(device)
#             metric_distance = model(metric_single)
#             # metric_distance = F.pairwise_distance(metric_embedding[:, 0] + metric_embedding[:, 1],
#             #                                       metric_embedding[:, 2], p=2)
#             metric_total_matrix = np.argsort(metric_distance.data.cpu().numpy())
#             rank = np.where(metric_total_matrix == x)[0][0]
#             self.total_rank.append(rank)
#         self.total_rank_numpy = np.array(self.total_rank)
#         self.raw_meanrank = np.mean(self.total_rank_numpy)
#         self.raw_hitatten = np.sum(self.total_rank_numpy <= 9) / len(metric_dataset) * 100
#         self.raw_MRR=1
#         pass