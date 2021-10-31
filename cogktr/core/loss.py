# import random
# import torch.nn as nn
# import numpy as np
# import torch
# import torch.nn.functional as F
# class MarginLoss(nn.Module):
#     def __init__(self,entity_dict_len):
#         super(MarginLoss, self).__init__()
#         self.entity_dict_len=entity_dict_len
#         pass
#
#     def create_negtive_sample(self,batch_sample):
#         batch_sample_negtive=batch_sample.clone().detach()
#         for i in range(len(batch_sample_negtive[:,0])):
#             if(random.random()<0.5):
#                 batch_sample_negtive[i][0]=np.random.randint(0,self.entity_dict_len)
#             else:
#                 batch_sample_negtive[i][2]=np.random.randint(0,self.entity_dict_len)
#         return batch_sample_negtive
#
#     def forward(self,positive_item,model):
#         negtive_item=self.create_negtive_sample(positive_item)
#
#         positive_item_head=torch.unsqueeze(model.entity_embedding(positive_item[:,0]), 1)
#         positive_item_relation=torch.unsqueeze(model.relation_embedding(positive_item[:,1]), 1)
#         positive_item_tail=torch.unsqueeze(model.entity_embedding(positive_item[:,2]), 1)
#         negtive_item_head=torch.unsqueeze(model.entity_embedding(negtive_item[:,0]), 1)
#         negtive_item_relation=torch.unsqueeze(model.relation_embedding(negtive_item[:,1]), 1)
#         negtive_item_tail=torch.unsqueeze(model.entity_embedding(negtive_item[:,2]), 1)
#
#         positive_item_head= F.normalize(positive_item_head, p=model.L, dim=2)
#         positive_item_tail= F.normalize(positive_item_tail, p=model.L, dim=2)
#         negtive_item_head= F.normalize(negtive_item_head, p=model.L, dim=2)
#         negtive_item_tail= F.normalize(negtive_item_tail, p=model.L, dim=2)
#
#         positive_item_distance=model.distance(positive_item_head+positive_item_relation,positive_item_tail)
#         negtive_item_distance=model.distance(negtive_item_head+negtive_item_relation,negtive_item_tail)
#
#         output=torch.sum(F.relu(model.margin+positive_item_distance-negtive_item_distance))/(positive_item.shape[0])
#         return output

########################################################################################################################
import numpy as np
import random
import torch
import torch.nn.functional as F
class MarginLoss:
    def __init__(self,margin):
        self.margin=margin
        pass

    def __call__(self,positive_batch,negtive_batch):
        positive_distance=F.pairwise_distance(positive_batch[:,0]+positive_batch[:,1],positive_batch[:,2],p=2)
        negtive_distance=F.pairwise_distance(negtive_batch[:,0]+negtive_batch[:,1],negtive_batch[:,2],p=2)
        output=torch.sum(F.relu(self.margin+positive_distance-negtive_distance))/(positive_batch.shape[0])
        return output
