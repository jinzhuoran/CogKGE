import torch


def description(func):
    def inner(*args, **kwargs):
        current_device = "cuda:%s" % (torch.cuda.current_device())
        head_embedding, relation_embedding, tail_embedding = func(*args, **kwargs)
        triplet_idx = args[1]
        node_lut = args[2]
        pre_training_model = args[5]
        head_token = node_lut.token[triplet_idx[:, 0]].to(current_device)
        tail_token = node_lut.token[triplet_idx[:, 2]].to(current_device)
        head_mask = node_lut.mask[triplet_idx[:, 0]].to(current_device)
        tail_mask = node_lut.mask[triplet_idx[:, 2]].to(current_device)
        head_embedding = pre_training_model(head_token, attention_mask=head_mask).pooler_output
        tail_embedding = pre_training_model(tail_token, attention_mask=tail_mask).pooler_output
        return head_embedding, relation_embedding, tail_embedding

    return inner
