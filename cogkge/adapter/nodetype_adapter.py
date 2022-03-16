import torch

# def nodetype(func):
#     def inner(*args, **kwargs):
#         current_device = "cuda:%s" % (torch.cuda.current_device())
#         head_embedding, relation_embedding, tail_embedding = func(*args, **kwargs)
#         triplet_idx = args[1]
#         node_lut = args[2]
#         nodetype_transfer_matrix = args[3]
#         embedding_dim = args[4]
#         head_type = node_lut.type[triplet_idx[:, 0]].to(current_device)
#         tail_type = node_lut.type[triplet_idx[:, 2]].to(current_device)
#         head_embedding = torch.bmm(torch.unsqueeze(head_embedding, dim=1),
#                                    nodetype_transfer_matrix(head_type).view(-1, embedding_dim, embedding_dim)).squeeze()
#         tail_embedding = torch.bmm(torch.unsqueeze(tail_embedding, dim=1),
#                                    nodetype_transfer_matrix(tail_type).view(-1, embedding_dim, embedding_dim)).squeeze()
#         return head_embedding, relation_embedding, tail_embedding
#
#     return inner

import torch.nn as nn


def nodetype(func):
    def inner(*args, **kwargs):
        h_embedding, r_embedding, t_embedding = func(*args, **kwargs)
        model = args[0]
        h_type = kwargs["batch"]["h_type"].to(model.model_device)
        t_type = kwargs["batch"]["t_type"].to(model.model_device)
        if not model.init_type_adapter:
            model.nodetype_transfer = nn.Embedding(num_embeddings=model.type_dict_len,
                                                   embedding_dim=model.embedding_dim * model.embedding_dim).to(
                model.model_device)
            model.init_type_adapter = True
        h_embedding = torch.bmm(torch.unsqueeze(h_embedding, dim=1),
                                model.nodetype_transfer(h_type).view(-1, model.embedding_dim,
                                                                     model.embedding_dim)).squeeze()
        t_embedding = torch.bmm(torch.unsqueeze(t_embedding, dim=1),
                                model.nodetype_transfer(t_type).view(-1, model.embedding_dim,
                                                                     model.embedding_dim)).squeeze()
        return h_embedding, r_embedding, t_embedding

    return inner
