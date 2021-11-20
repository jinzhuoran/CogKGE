import torch
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm

class Link_Prediction:
    def __init__(self, entity_dict_len):
        self.entity_dict_len = entity_dict_len
        self.name = "Link_Prediction"
        self.total_rank = None
        self.total_rank_numpy = None

        self.MRR = []
        self.MRR_numpy = None

        self.raw_meanrank = None
        self.raw_hitatten = None
        self.raw_MRR = None

    def __call__(self, model, metric_dataset,device):
        metric_loader = Data.DataLoader(dataset=metric_dataset, batch_size=1, shuffle=False)
        self.total_rank = list()
        for step, batch_sample in enumerate(tqdm(metric_loader)):
            # replace the tail
            metric_single = copy.deepcopy(batch_sample)
            x = metric_single[:, 2][0].numpy()
            metric_single = metric_single.expand(self.entity_dict_len, 3)
            metric_single = metric_single[:, :2]
            new_tail = torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
            metric_single = torch.cat([metric_single,new_tail],dim=1)
            metric_single = metric_single.to(device)
            metric_distance = model.get_score(metric_single)
            metric_total_matrix = np.argsort(metric_distance.data.cpu().numpy())
            rank_tail = np.where(metric_total_matrix == x)[0][0]
            self.total_rank.append(rank_tail)
            self.MRR.append(1/(rank_tail+1))

            # replace the head
            metric_single = batch_sample
            x = metric_single[:, 0][0].numpy()
            metric_single = metric_single.expand(self.entity_dict_len, 3)
            metric_single = metric_single[:, 1:]
            new_head = torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
            metric_single = torch.cat([new_head,metric_single],dim=1)
            metric_single = metric_single.to(device)
            metric_distance = model.get_score(metric_single)
            metric_total_matrix = np.argsort(metric_distance.data.cpu().numpy())
            rank_head = np.where(metric_total_matrix == x)[0][0]
            self.total_rank.append(rank_head)
            self.MRR.append(1/(rank_head+1))
            

        self.total_rank_numpy = np.array(self.total_rank)
        self.MRR_numpy = np.array(self.MRR)
        self.raw_meanrank = np.mean(self.total_rank_numpy)
        self.raw_hitatten = np.sum(self.total_rank_numpy <= 9) / (2*len(metric_dataset)) * 100
        self.raw_MRR = np.mean(self.MRR_numpy)

# need to be modified!
class LinkRotatePrediction:
    def __init__(self, entity_dict_len):
        self.entity_dict_len = entity_dict_len
        self.name = "Link_Prediction"
        self.total_rank = None
        self.total_rank_numpy = None
        self.raw_meanrank = None
        self.raw_hitatten = None

    def __call__(self, model, metric_dataset,device):
        metric_loader = Data.DataLoader(dataset=metric_dataset, batch_size=1, shuffle=False)
        self.total_rank = list()
        for step, metric_single in enumerate(metric_loader):
            x = metric_single[:, 2][0].numpy()
            metric_single = metric_single.expand(self.entity_dict_len, 3)
            metric_single = metric_single[:, :2]
            new_tail = torch.unsqueeze(torch.arange(0, self.entity_dict_len), dim=1)
            # metric_single = torch.hstack((metric_single, new_tail))
            metric_single = torch.cat([metric_single,new_tail],dim=1)
            metric_single = metric_single.to(device)
            metric_embedding = model(metric_single)
            metric_distance = F.pairwise_distance(metric_embedding[:, 0] * metric_embedding[:, 1],
                                                  metric_embedding[:, 2], p=2)
            metric_total_matrix = np.argsort(metric_distance.data.cpu().numpy())
            rank = np.where(metric_total_matrix == x)[0][0]
            self.total_rank.append(rank)
        self.total_rank_numpy = np.array(self.total_rank)
        self.raw_meanrank = np.mean(self.total_rank_numpy)
        self.raw_hitatten = np.sum(self.total_rank_numpy <= 9) / len(metric_dataset) * 100
        pass