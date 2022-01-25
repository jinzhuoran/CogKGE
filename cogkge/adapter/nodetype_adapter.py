import torch


def nodetype(func):
    def inner(*args, **kwargs):
        current_device = "cuda:%s" % (torch.cuda.current_device())
        head_embedding, relation_embedding, tail_embedding = func(*args, **kwargs)
        triplet_idx = args[1]
        node_lut = args[2]
        nodetype_transfer_matrix = args[3]
        embedding_dim = args[4]
        head_type = node_lut.type[triplet_idx[:, 0]].to(current_device)
        tail_type = node_lut.type[triplet_idx[:, 2]].to(current_device)
        head_embedding = torch.bmm(torch.unsqueeze(head_embedding, dim=1),
                                   nodetype_transfer_matrix(head_type).view(-1, embedding_dim, embedding_dim)).squeeze()
        tail_embedding = torch.bmm(torch.unsqueeze(tail_embedding, dim=1),
                                   nodetype_transfer_matrix(tail_type).view(-1, embedding_dim, embedding_dim)).squeeze()
        return head_embedding, relation_embedding, tail_embedding

    return inner
