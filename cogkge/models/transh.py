import torch
import torch.nn as nn
import torch.nn.functional as F


class TransH(nn.Module):

    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=50, p=2, epsilon=0.01):
        super(TransH, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.dim = embedding_dim
        self.p = p
        self.epsilon = epsilon
        self.name = "TransH"

        self.entity_embedding = nn.Embedding(self.entity_dict_len, self.dim)
        self.relation_embedding = nn.Embedding(self.relation_dict_len, self.dim)
        self.w_vector = nn.Embedding(self.relation_dict_len, self.dim)

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.w_vector.weight.data)

        self.head_batch_embedding = None
        self.relation_batch_embedding = None
        self.tail_batch_embedding = None
        self.w_r_batch_embedding = None

    def get_score(self, triplet_idx):
        # (batch,3)

        output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)

        return score

    def get_embedding(self, triplet_idx):
        return self._forward(triplet_idx)

    def forward(self, triplet_idx):
        return self.get_score(triplet_idx)

    def _forward(self, triplet_idx):
        # (batch,d)
        head_embedding = self.entity_embedding(triplet_idx[:, 0])
        relation_embedding = self.relation_embedding(triplet_idx[:, 1])
        tail_embedding = self.entity_embedding(triplet_idx[:, 2])
        w_r = self.w_vector(triplet_idx[:, 1])  # (batch,d)
        w_r = F.normalize(w_r, p=2, dim=1)

        self.head_batch_embedding = head_embedding
        self.relation_batch_embedding = relation_embedding
        self.tail_batch_embedding = tail_embedding
        self.w_r_batch_embedding = w_r

        head_embedding = head_embedding - torch.sum(head_embedding * w_r, 1, True) * w_r
        tail_embedding = tail_embedding - torch.sum(tail_embedding * w_r, 1, True) * w_r

        # head_embedding = F.normalize(head_embedding, p=2, dim=1)
        relation_embedding = F.normalize(relation_embedding, p=2, dim=1)
        # tail_embedding = F.normalize(tail_embedding, p=2, dim=1)

        head_embedding = torch.unsqueeze(head_embedding, 1)
        relation_embedding = torch.unsqueeze(relation_embedding, 1)
        tail_embedding = torch.unsqueeze(tail_embedding, 1)
        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding

        return output

    def get_penalty(self):
        # constraint_1=torch.sum(nn.ReLU(inplace=False)(self.head_batch_embedding ** 2-1/len(self.head_batch_embedding)))
        # constraint_2=torch.sum(nn.ReLU(inplace=False)(self.tail_batch_embedding ** 2-1/len(self.tail_batch_embedding)))
        # constraint_3=sum(sum((self.relation_batch_embedding*self.w_r_batch_embedding)** 2)/sum(self.relation_batch_embedding**2)-self.epsilon**2)
        # penalty=constraint_1+constraint_2+constraint_3
        penalty = (torch.mean(self.head_batch_embedding ** 2) +
                   torch.mean(self.relation_batch_embedding ** 2) +
                   torch.mean(self.tail_batch_embedding ** 2) +
                   torch.mean(self.w_r_batch_embedding ** 2)) / 4
        return penalty

# class TransH(nn.Module):
#
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim=100, p_norm=1, mode='head_batch'):
#         super(TransH, self).__init__()
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.dim = embedding_dim
#         self.p_norm = p_norm
#         self.name = "TransH"
#         self.mode = mode
#         self.entity_embedding = nn.Embedding(self.entity_dict_len, self.dim)
#         self.relation_embedding = nn.Embedding(self.relation_dict_len, self.dim)
#         self.norm_vector = nn.Embedding(self.relation_dict_len, self.dim)
#
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#         nn.init.xavier_uniform_(self.norm_vector.weight.data)
#
#     def get_score(self, triplet_idx):
#         #(batch,3)
#         h, r, t = self._forward(triplet_idx)
#         h = F.normalize(h, 2, -1)
#         r = F.normalize(r, 2, -1)
#         t = F.normalize(t, 2, -1)
#         if self.mode != 'normal':
#             h = h.view(-1, r.shape[0], h.shape[-1])
#             t = t.view(-1, r.shape[0], t.shape[-1])
#             r = r.view(-1, r.shape[0], r.shape[-1])
#         if self.mode == 'head_batch':
#             score = h + (r - t)
#         else:
#             score = (h + r) - t
#         score = torch.norm(score, self.p_norm, -1).flatten()
#         return score
#
#     def get_embedding(self, triplet_idx):
#         return self._forward(triplet_idx)
#
#     def forward(self, triplet_idx):
#         return self.get_score(triplet_idx)
#
#     def _forward(self, triplet_idx):
#         #(batch,d)
#         head_embedding = self.entity_embedding(triplet_idx[:, 0])
#         relation_embedding = self.relation_embedding(triplet_idx[:, 1])
#         tail_embedding = self.entity_embedding(triplet_idx[:, 2])
#
#         r_norm = self.norm_vector(triplet_idx[:, 1])  #(batch,d)
#         head_embedding = self._transfer(head_embedding, r_norm)
#         tail_embedding = self._transfer(tail_embedding, r_norm)
#         return head_embedding, relation_embedding, tail_embedding
#
#     def _transfer(self, e, norm):
#         norm = F.normalize(norm, p=2, dim=-1)
#         if e.shape[0] != norm.shape[0]:
#             e = e.view(-1, norm.shape[0], e.shape[-1])
#             norm = norm.view(-1, norm.shape[0], norm.shape[-1])
#             e = e - torch.sum(e * norm, -1, True) * norm
#             return e.view(-1, e.shape[-1])
#         else:
#             return e - torch.sum(e * norm, -1, True) * norm


#
#
# class TransH(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, embedding_dim, p=2.0):
#         super(TransH, self).__init__()
#         self.name = "TransH"
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.entity_embedding = nn.Embedding(entity_dict_len, embedding_dim)
#         self.translation_embedding = nn.Embedding(relation_dict_len, embedding_dim)
#         self.norm_vector = nn.Embedding(relation_dict_len, embedding_dim)
#         self.p = p
#
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.translation_embedding.weight.data)
#         nn.init.xavier_uniform_(self.norm_vector.weight.data)
#
#     def forward(self, sample):
#         batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
#         h = self.entity_embedding(batch_h)
#         d_r = self.translation_embedding(batch_r)
#         t = self.entity_embedding(batch_t)
#
#         w_r = self.norm_vector(batch_r)
#         w_r = F.normalize(w_r, p=2.0, dim=-1)
#
#         h_vertical = h - torch.sum(h * w_r, -1, True) * w_r
#         t_vertical = t - torch.sum(t * w_r, -1, True) * w_r
#
#         h_vertical = F.normalize(h_vertical)
#         t_vertical = F.normalize(t_vertical)
#
#         score = torch.norm(h_vertical + d_r - t_vertical, dim=-1, p=self.p)
#         return score
#
#     # def transfer(self,e,norm):
#     #     norm = F.normalize(norm,p=2.0,dim=-1)
#     #     return e - torch.sum(e * norm,-1,True) * norm
#
#     # def get_score(self,sample):
#     #     output = self._forward(sample)
#     #     score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
#     #     return score  # (batch,)
#
#     # def get_embedding(self,sample):
#     #     return self._forward(sample)
#
#     # def forward(self,sample):
#     #     return self.get_score(sample)
#
#     # def _forward(self, sample):  # sample:(batch,3)
#     #     batch_h,batch_r,batch_t =  sample[:, 0], sample[:, 1], sample[:, 2]
#     #     h = self.entity_embedding(batch_h)
#     #     r = self.relation_embedding(batch_r)
#     #     t = self.entity_embedding(batch_t)
#
#     #     r_norm = self.norm_vector(batch_r)
#
#     #     h = self.transfer(h,r_norm)
#     #     t = self.transfer(t,r_norm)
#
#     #     h = F.normalize(h, p=2.0,dim=-1)
#     #     r = F.normalize(r, p=2.0, dim=-1)
#     #     t = F.normalize(t, p=2.0, dim=-1)
#
#     #     h = torch.unsqueeze(h,1)
#     #     r = torch.unsqueeze(r,1)
#     #     t = torch.unsqueeze(t,1)
#
#     #     return torch.cat((h,r,t),dim=1) # return:(batch,3,embedding_dim)
