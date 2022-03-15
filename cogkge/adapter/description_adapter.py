import torch


# def description(func):
#     def inner(*args, **kwargs):
#         current_device = "cuda:%s" % (torch.cuda.current_device())
#         head_embedding, relation_embedding, tail_embedding = func(*args, **kwargs)
#         triplet_idx = args[1]
#         node_lut = args[2]
#         pre_training_model = args[5]
#         head_token = node_lut.token[triplet_idx[:, 0]].to(current_device)
#         tail_token = node_lut.token[triplet_idx[:, 2]].to(current_device)
#         head_mask = node_lut.mask[triplet_idx[:, 0]].to(current_device)
#         tail_mask = node_lut.mask[triplet_idx[:, 2]].to(current_device)
#         head_embedding = pre_training_model(head_token, attention_mask=head_mask).pooler_output
#         tail_embedding = pre_training_model(tail_token, attention_mask=tail_mask).pooler_output
#         return head_embedding, relation_embedding, tail_embedding
#
#     return inner

import torch.nn as nn
from transformers import RobertaModel
def description_adapter(func):
    def inner(*args, **kwargs):
        h_embedding, r_embedding, t_embedding = func(*args, **kwargs)
        model=args[0]
        h_token=kwargs["data"][3]
        t_token=kwargs["data"][4]
        h_mask=kwargs["data"][5]
        t_mask=kwargs["data"][6]
        if not model.init_description_adapter:
            model.pre_training_model_name = "roberta-base"
            model.pre_training_model = RobertaModel.from_pretrained(model.pre_training_model_name).to(model.model_device)
            model.out_dim=model.pre_training_model.pooler.dense.out_features
            model.init_description_adapter=True
        h_embedding=model.pre_training_model(h_token,h_mask).pooler_output
        t_embedding=model.pre_training_model(t_token,t_mask).pooler_output
        return h_embedding, r_embedding, t_embedding
    return inner
