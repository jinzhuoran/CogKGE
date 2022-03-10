import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self,model_name="BaseModel",penalty_weight=0.0):
        super().__init__()
        self.model_loss=None
        self.model_metric=None
        self.model_negative_sampler=None
        self.model_name=model_name
        self.penalty_weight=penalty_weight

    def set_model_config(self,model_loss=None,model_metric=None,model_negative_sampler=None):
        #设置模型使用的metric和loss
        self.model_loss=model_loss
        self.model_metrci=model_metric
        self.model_negative_sampler=model_negative_sampler

    def _reset_param(self):
        #重置参数
        pass

    def forward(self,data):
        #前向传播
        data=self.get_batch(data)
        pass

    def get_realation_embedding(self,relation_ids):
        #得到关系的embedding
        pass

    def get_entity_embedding(self,entity_ids):
        #得到实体的embedding
        pass

    def get_triplet_embedding(self,tri):
        #得到三元组的embedding
        pass

    def get_batch(self,data):
        #得到一个batch的数据
        return data

    def loss(self,data):
        #计算损失
        pass

    def negative_sample(self,data):
        #负采样
        pass

    def penalty(self):
        # 正则项
        penalty_loss=torch.tensor(0.0)
        for param in self.parameters():
            penalty_loss+=torch.sum(param**2)
        return self.penalty_weight*penalty_loss

    def metric(self,data):
        #模型评价
        pass