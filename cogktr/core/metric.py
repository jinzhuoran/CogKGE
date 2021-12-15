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

# import math
# import copy
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import prettytable as pt
# import torch.utils.data as Data
# from collections import defaultdict
#
# class Link_Prediction(object):
#     def __init__(self,
#                  batch_size,
#                  reverse,
#                  link_prediction_raw=True,
#                  link_prediction_filt=False,
#                  triple_classification=False):
#         self.batch_size=batch_size
#         self.reverse=reverse
#         self.link_prediction_raw=link_prediction_raw
#         self.link_prediction_filt=link_prediction_filt
#         self.triple_classification=triple_classification
#
#         self._nodetype_columnindex_dict={"head":0,"tail":2}
#         self._create_correct_triplet_dict_flag=True
#
#     def _init_cogkr_metric(self):
#         torch.cuda.empty_cache()
#         self._raw_rank=-1
#         self._filt_rank=-1
#         self._raw_rank_list = list()
#         self._filt_rank_list = list()
#         self._raw_rank=-1
#         self._filt_rank=-1
#         self._Raw_MR = -1
#         self._Raw_MRR = -1
#         self._Raw_Hits1 = -1
#         self._Raw_Hits3 = -1
#         self._Raw_Hits10 = -1
#         self._Filt_MR =-1
#         self._Filt_MRR = -1
#         self._Filt_Hits1 = -1
#         self._Filt_Hits3 = -1
#         self._Filt_Hits10 = -1
#
#     def _create_correct_node_dict(self,dataset):
#         node_dict=defaultdict(dict)
#         for index in range(len(dataset)):
#             r_t=tuple(dataset.data_numpy[index][1:])
#             h_r=tuple(dataset.data_numpy[index][:2])
#             h=torch.tensor(dataset.data_numpy[index][0])
#             t=torch.tensor(dataset.data_numpy[index][2])
#             if r_t not in node_dict["head"]:
#                 node_dict["head"].setdefault(r_t,[])
#             node_dict["head"][r_t].append(h)
#             if h_r not in node_dict["tail"]:
#                 node_dict["tail"].setdefault(h_r,[])
#             node_dict["tail"][h_r].append(t)
#         return node_dict
#     def _init_correct_node_dict(self):
#         if self._train_dataset!=None:
#             self._correct_train_node_dict=self._create_correct_node_dict(self._train_dataset)
#         if self._valid_dataset!=None:
#             self._correct_valid_node_dict=self._create_correct_node_dict(self._valid_dataset)
#         if self._test_dataset!=None:
#             self._correct_test_node_dict=self._create_correct_node_dict(self._test_dataset)
#         self._create_correct_triplet_dict_flag=False
#
#
#
#     def _generate_expanded_triplet(self,single_triplet,entity_type):
#         expanded_triplet=single_triplet.expand(self._node_dict_len, 3)
#         if entity_type=="head":
#             original_r_t = expanded_triplet[:, 1:]
#             expanded_node = torch.unsqueeze(torch.arange(0,self._node_dict_len), dim=1).to(self._device)
#             expanded_triplet = torch.cat([expanded_node, original_r_t], dim=1)
#         if entity_type=="tail":
#             original_h_r=expanded_triplet[:, :2]
#             expanded_node= torch.unsqueeze(torch.arange(0, self._node_dict_len), dim=1).to(self._device)
#             expanded_triplet= torch.cat([original_h_r,expanded_node], dim=1)
#         return expanded_triplet
#
#     def _caculate_rank(self,single_sample,entity_type):
#         single_triplet=copy.deepcopy(single_sample)
#         column_index=self._nodetype_columnindex_dict[entity_type]
#         correct_id=single_triplet[:,column_index]
#         expanded_triplet=self._generate_expanded_triplet(single_triplet,entity_type)
#
#         score_list = list()
#         for i in range(math.ceil(self._node_dict_len / self.batch_size)):
#             start = i * self.batch_size
#             end = min((i + 1) * self.batch_size, self._node_dict_len)
#             data_input = expanded_triplet[start:end, :]
#             with torch.no_grad():
#                 result=self._model(data_input)
#                 score_list.append(result)
#         sorted_score_rank= torch.argsort(torch.cat(score_list,dim=0))
#         correct_triplet_rank= torch.where(sorted_score_rank == correct_id)[0].item()+1
#
#         if self.link_prediction_raw:
#             raw_correct_triplet_rank=self._node_dict_len+1-correct_triplet_rank if self.reverse else correct_triplet_rank
#             self._raw_rank_list.append(raw_correct_triplet_rank)
#
#
#         if self.link_prediction_filt:
#             valid_correct_node_index_list=list()
#             train_correct_node_index_list=list()
#             arr=1 if entity_type=="head" else 0
#             sorted_score_rank=sorted_score_rank.cpu()
#             triplet_key=tuple(single_triplet[0][arr:arr+2].cpu().numpy())
#             if triplet_key in self._correct_valid_node_dict[entity_type].keys():
#                 valid_correct_node_index_list=[torch.where(sorted_score_rank == x)[0].item()+1 for x in self._correct_valid_node_dict[entity_type][triplet_key]]
#             if triplet_key in self._correct_train_node_dict[entity_type].keys():
#                 train_correct_node_index_list=[torch.where(sorted_score_rank == x)[0].item()+1 for x in self._correct_train_node_dict[entity_type][triplet_key]]
#             valid_correct_before_num=torch.sum(torch.tensor(valid_correct_node_index_list)<correct_triplet_rank)
#             train_correct_before_num=torch.sum(torch.tensor(train_correct_node_index_list)<correct_triplet_rank)
#             correct_triplet_num=valid_correct_before_num+train_correct_before_num
#             filt_correct_triplet_rank=correct_triplet_rank-correct_triplet_num
#             self._filt_rank_list.append(filt_correct_triplet_rank)
#
#
#
#     def caculate(self,
#                  device,
#                  model,
#                  total_epoch,
#                  current_epoch,
#                  metric_type,
#                  metric_dataset,
#                  node_dict_len,
#                  model_name,
#                  logger,
#                  writer,
#                  train_dataset=None,
#                  valid_dataset=None,
#                  test_dataset=None):
#         self._device=device
#         self._model=model
#         self._total_epoch=total_epoch
#         self._current_epoch=current_epoch
#         self._metric_type=metric_type
#         self._metric_dataset=metric_dataset
#         self._node_dict_len=node_dict_len
#         self._model_name=model_name
#         self._logger=logger
#         self._writer=writer
#         self._train_dataset=train_dataset
#         self._valid_dataset=valid_dataset
#         self._test_dataset=test_dataset
#
#         self._init_cogkr_metric()
#         if self.link_prediction_filt and self._create_correct_triplet_dict_flag:
#             self._init_correct_node_dict()
#
#         metric_loader = Data.DataLoader(dataset=metric_dataset, batch_size=1, shuffle=False)
#         for step, single_sample in enumerate(tqdm(metric_loader)):
#             single_sample=single_sample.to(self._device)
#             self._caculate_rank(single_sample,"tail")
#             self._caculate_rank(single_sample,"head")
#         if self.link_prediction_raw:
#             self._raw_rank=torch.tensor(self._raw_rank_list,dtype= torch.float64)
#             self._Raw_Hits1 = round((torch.sum(self._raw_rank <= 1) / (2 * len(metric_dataset)) * 100).item(), 3)
#             self._Raw_Hits3  = round((torch.sum(self._raw_rank <= 3) / (2 * len(metric_dataset)) * 100).item(), 3)
#             self._Raw_Hits10 = round((torch.sum(self._raw_rank<= 10) / (2 * len(metric_dataset)) * 100).item(), 3)
#             self._Raw_MR= round(torch.mean(self._raw_rank).item(), 3)
#             self._Raw_MRR  = round(torch.mean(1/self._raw_rank).item(), 3)
#         if self.link_prediction_filt:
#             self._filt_rank=torch.tensor(self._filt_rank_list,dtype= torch.float64)
#             self._Filt_Hits1 = round((torch.sum(self._filt_rank <= 1) / (2 * len(metric_dataset)) * 100).item(), 3)
#             self._Filt_Hits3  = round((torch.sum(self._filt_rank <= 3) / (2 * len(metric_dataset)) * 100).item(), 3)
#             self._Filt_Hits10 = round((torch.sum(self._filt_rank<= 10) / (2 * len(metric_dataset)) * 100).item(), 3)
#             self._Filt_MR= round(torch.mean(self._filt_rank).item(), 3)
#             self._Filt_MRR  = round(torch.mean(1/self._filt_rank).item(), 3)
#
#
#     def get_Raw_Rank(self):
#         return self._raw_rank
#     def get_Filt_Rank(self):
#         return self._filt_rank
#     def get_Raw_MR(self):
#         return self._Raw_MR
#     def get_Raw_MRR(self):
#         return self._Raw_MRR
#     def get_Raw_Hits1(self):
#         return self._Raw_Hits1
#     def get_Raw_Hits3(self):
#         return self._Raw_Hits3
#     def get_Raw_Hits10(self):
#         return self._Raw_Hits10
#     def get_Filt_MR(self):
#         return self._Filt_MR
#     def get_Filt_MRR(self):
#         return self._Filt_MRR
#     def get_Filt_Hits1(self):
#         return self._Filt_Hits1
#     def get_Filt_Hits3(self):
#         return self._Filt_Hits3
#     def get_Filt_Hits10(self):
#         return self._Filt_Hits10
#     def print_table(self):
#         tb = pt.PrettyTable()
#         tb.field_names = [self._model_name,
#                           "Hits@1",
#                           "Hits@3",
#                           "Hits@10",
#                           "MR",
#                           "MRR"]
#         if self.link_prediction_raw:
#             tb.add_row(["Raw",
#                         self._Raw_Hits1,
#                         self._Raw_Hits3,
#                         self._Raw_Hits10,
#                         self._Raw_MR,
#                         self._Raw_MRR])
#         if self.link_prediction_filt:
#             tb.add_row(["Filt",
#                         self._Filt_Hits1,
#                         self._Filt_Hits3,
#                         self._Filt_Hits10,
#                         self._Filt_MR,
#                         self._Filt_MRR])
#         print(tb)
#
#     def log(self):
#         if self.link_prediction_raw:
#             self._logger.info("Epoch {}/{}  Raw_Hits@1:{}   Raw_Hits@3:{}   Raw_Hits@10:{}   Raw_MR:{}   Raw_MRR:{}".format(
#                 self._current_epoch, self._total_epoch, self._Raw_Hits1, self._Raw_Hits3, self._Raw_Hits10, self._Raw_MR, self._Raw_MRR))
#         if self.link_prediction_filt:
#             self._logger.info("Epoch {}/{}  Filt_Hits@1:{}   Filt_Hits@3:{}   Filt_Hits@10:{}   Filt_MR:{}   Filt_MRR:{}   ".format(
#                 self._current_epoch, self._total_epoch, self._Filt_Hits1, self._Filt_Hits3, self._Filt_Hits10, self._Filt_MR, self._Filt_MRR))
#     def write(self):
#         if self.link_prediction_raw:
#             self._writer.add_scalars("Hits@1", {"{}_Raw_Hits@1".format(self._metric_type): self._Raw_Hits1}, self._current_epoch)
#             self._writer.add_scalars("Hits@3", {"{}_Raw_Hits@3".format(self._metric_type): self._Raw_Hits3}, self._current_epoch)
#             self._writer.add_scalars("Hits@10", {"{}_Raw_Hits@10".format(self._metric_type): self._Raw_Hits10}, self._current_epoch)
#             self._writer.add_scalars("MR", {"{}_Raw_MR".format(self._metric_type): self._Raw_MR}, self._current_epoch)
#             self._writer.add_scalars("MRR", {"{}_Raw_MRR".format(self._metric_type): self._Raw_MRR}, self._current_epoch)
#         if self.link_prediction_filt:
#             self._writer.add_scalars("Hits@1", {"{}_Filt_Hits@1".format(self._metric_type): self._Filt_Hits1}, self._current_epoch)
#             self._writer.add_scalars("Hits@3", {"{}_Filt_Hits@3".format(self._metric_type): self._Filt_Hits3},self._current_epoch)
#             self._writer.add_scalars("Hits@10", {"{}_Filt_Hits@10".format(self._metric_type): self._Filt_Hits10},self._current_epoch)
#             self._writer.add_scalars("MR", {"{}_Filt_MR".format(self._metric_type): self._Filt_MR}, self._current_epoch)
#             self._writer.add_scalars("MRR", {"{}_Filt_MRR".format(self._metric_type): self._Filt_MRR}, self._current_epoch)

