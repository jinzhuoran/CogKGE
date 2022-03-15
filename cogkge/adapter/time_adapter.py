import torch


# def time(func):
#     def inner(*args, **kwargs):
#         current_device = "cuda:%s" % (torch.cuda.current_device())
#         head_embedding, relation_embedding, tail_embedding = func(*args, **kwargs)
#         triplet_idx = args[1]
#         time_transfer_matrix = args[6]
#         start_time = time_transfer_matrix(triplet_idx[:, 3]).to(current_device)
#         end_time = time_transfer_matrix(triplet_idx[:, 4]).to(current_device)
#         head_embedding = torch.cat((head_embedding, start_time, end_time), dim=1)
#         relation_embedding = torch.cat((relation_embedding, start_time, end_time), dim=1)
#         tail_embedding = torch.cat((tail_embedding, start_time, end_time), dim=1)
#         return head_embedding, relation_embedding, tail_embedding
#
#     return inner

import torch.nn as nn
def time_adapter(func):
    def inner(*args, **kwargs):
        h_embedding, r_embedding, t_embedding = func(*args, **kwargs)
        model=args[0]
        start=kwargs["data"][3]
        end=kwargs["data"][4]
        if not model.init_time_adapter:
            model.start_time_transfer=nn.Embedding(num_embeddings=model.time_dict_len, embedding_dim=10).to(model.model_device)
            model.end_time_transfer = nn.Embedding(num_embeddings=model.time_dict_len, embedding_dim=10).to(model.model_device)
            model.init_time_adapter=True
        strat_embedding=model.start_time_transfer(start)
        end_embedding=model.end_time_transfer(end)
        h_embedding=torch.cat((h_embedding,strat_embedding,end_embedding),dim=1)
        r_embedding=torch.cat((r_embedding,strat_embedding,end_embedding),dim=1)
        t_embedding=torch.cat((t_embedding,strat_embedding,end_embedding),dim=1)
        return h_embedding, r_embedding, t_embedding
    return inner



