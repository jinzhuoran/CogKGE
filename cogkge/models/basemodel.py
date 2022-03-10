import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_loss=None
        self.model_metric=None

    def _set_model_config(self,model_loss,model_metric):
        #设置模型使用的metric和loss
        self.model_loss=model_loss
        self.model_metrci=model_metric

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

    def get_batch(self,data):
        #得到一个batch的数据
        return data

    def loss(self,data):
        #计算损失
        pass

    def penalty(self):
        #正则项
        pass

    def metric(self,data):
        #模型评价
        pass