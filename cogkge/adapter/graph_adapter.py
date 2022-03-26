import torch


def graph_adapter(func):
    def inner(*args, **kwargs):
        # head_embedding, relation_embedding, tail_embedding = func(*args, **kwargs)
        model = args[0]
        sample = kwargs["data"]
        # sample = args[1]
        batch_h,batch_r,batch_t = sample[0],sample[1],sample[2]
        x = model.conv1(model.init_embed, model.edge_index)

        head_embedding = torch.index_select(x, 0, batch_h)
        tail_embedding = torch.index_select(x, 0, batch_t)
        relation_embedding = torch.index_select(model.init_rel, 0, batch_r)

        return head_embedding, relation_embedding, tail_embedding

    return inner
