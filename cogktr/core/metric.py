# import torch.nn as nn
# import numpy as np
# import torch
# import torch.nn.functional as F
# class MeanRank_HitAtTen(nn.Module):
#     def __init__(self,sample_num,test_epoch,entity2idx_len):
#         super(MeanRank_HitAtTen, self).__init__()
#         self.sample_num=sample_num
#         self.test_epoch=test_epoch
#         self.entity2idx_len=entity2idx_len
#         self.result_rank_numpy=None
#         self.result_rank_list=list()
#         self.mean_rank=None
#         self.hit_at_ten=None
#     def forward(self,data,model):
#         #刷新self.result_rank_list，否则会累积
#         self.result_rank_list=list()
#         for epoch in range(self.test_epoch):
#
#             random_test_batch_idx=np.random.randint(len(data)-1)
#
#             for step_test,test_batch_temp in enumerate(data):
#                 if step_test==random_test_batch_idx:
#                     test_batch_temp=test_batch_temp
#                     break
#
#             test_batch=test_batch_temp[:self.sample_num,:]
#
#
#             for i in range(self.sample_num-1):
#                 test_batch[i+1][2]=torch.tensor(np.random.randint(0,self.entity2idx_len))
#
#             output=model(test_batch.cuda())
#
#
#             distance_list=list()
#             for i in range(self.sample_num):
#                 distance_list.append(F.pairwise_distance(output[i,0]+output[i,1], output[i,2], p=2)[0].cpu().data.numpy())
#
#
#             distance_numpy=np.array(distance_list)
#             distance_index=np.argsort(distance_numpy)
#             for i in range(self.sample_num):
#                 if distance_index[i]==0:
#                     self.result_rank_list.append(i)
#
#         self.result_rank_numpy=np.array(self.result_rank_list)
#         self.mean_rank=np.mean(self.result_rank_numpy)
#         self.hit_at_ten=np.sum(self.result_rank_numpy<=10)
#         return 0
########################################################################################################################
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler
class Link_Prediction:
    def __init__(self,entity_dict_len,sample_num,repeat_epoch):
        self.entity_dict_len=entity_dict_len
        self.sample_num=sample_num
        self.repeat_epoch=repeat_epoch
        self.name="Link_Prediction"
        self.total_rank_numpy=None
        self.meanrank=None
        self.hitatten=None

    def __call__(self,model,metric_dataset):
        metric_sampler  = RandomSampler(metric_dataset)
        metric_loader = Data.DataLoader(dataset=metric_dataset,sampler=metric_sampler,batch_size=self.sample_num)
        self.total_rank=list()
        for epoch in range(self.repeat_epoch):
            for step,metric_batch in enumerate(metric_loader):
                if step==0:
                    metric_batch=metric_batch.cuda()
                else:
                    break

                for i in range(self.sample_num-1):
                    metric_batch[i+1][2]=torch.tensor(np.random.randint(0,self.entity_dict_len))
                metric_batch_embedding=model(metric_batch)
                metric_distance=F.pairwise_distance(metric_batch_embedding[:,0]+metric_batch_embedding[:,1],
                                                    metric_batch_embedding[:,2],p=2)
                metric_total_matrix= np.argsort(metric_distance.data.cpu().numpy())
                rank=np.where(metric_total_matrix==0)[0][0]
                self.total_rank.append(rank)
        self.total_rank_numpy=np.array(self.total_rank)
        self.meanrank=np.mean(self.total_rank_numpy)
        self.hitatten=np.sum(self.total_rank_numpy<=9)/self.repeat_epoch*100
        pass