import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, model_name="BaseModel", penalty_weight=0.0):
        super().__init__()
        self.model_loss = None
        self.model_metric = None
        self.model_negative_sampler = None
        self.model_name = model_name
        self.penalty_weight = penalty_weight
        self.init_description_adapter = False
        self.init_type_adapter = False
        self.init_time_adapter = False
        self.init_graph_adapter = False
        self.time_dict_len = 0
        self.nodetype_dict_len = 0
        self.relationtype_dict_len = 0

    def set_model_config(self,
                         model_loss=None,
                         model_metric=None,
                         model_negative_sampler=None,
                         model_device="cpu",
                         time_dict_len=0,
                         nodetype_dict_len=0,
                         relationtype_dict_len=0):
        # 设置模型使用的metric和loss
        self.model_loss = model_loss
        self.model_metric = model_metric
        self.model_negative_sampler = model_negative_sampler
        self.model_device = model_device
        self.time_dict_len = time_dict_len
        self.nodetype_dict_len = nodetype_dict_len
        self.relationtype_dict_len = relationtype_dict_len

    def _reset_param(self):
        # 重置参数
        pass

    def forward(self, data):
        # 前向传播
        pass

    def get_realation_embedding(self, relation_ids):
        # 得到关系的embedding
        pass

    def get_entity_embedding(self, entity_ids):
        # 得到实体的embedding
        pass

    def get_triplet_embedding(self, data):
        # 得到三元组的embedding
        pass

    def loss(self, data):
        # 计算损失
        data = self.get_batch(data)
        pass

    def penalty(self,data):
        # 正则项
        penalty_loss = torch.tensor(0.0)
        return self.penalty_weight * penalty_loss

    def data_to_device(self, data):
        self.model_device = self.cal_gpu()
        for index, item in enumerate(data):
            data[index] = item.to(self.model_device)
        return data

    def cal_gpu(self):
        if isinstance(self, torch.nn.DataParallel):
            module = self.module
        else:
            module = self
        for submodule in module.children():
            if hasattr(submodule, "_parameters"):
                parameters = submodule._parameters
                if "weight" in parameters:
                    return parameters["weight"].device