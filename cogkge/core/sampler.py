import torch


class UnifNegativeSampler():
    def __init__(self, triples, entity_dict_len, relation_dict_len, device=torch.device('cuda:0')):
        # tensor(len,3)
        self.triples = triples
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.device = device

    def create_negative(self, batch_pos):
        # tensor(batch,3)
        batch_neg = batch_pos.clone()
        entity_number = torch.randint(self.entity_dict_len, (batch_neg.size()[0],)).to(self.device)
        mask = torch.rand(batch_neg.size()[0])
        head_mask = (mask > 0.5).to(self.device)
        tail_mask = (mask <= 0.5).to(self.device)
        batch_neg[head_mask, 0] = entity_number[head_mask].to(self.device)
        batch_neg[tail_mask, 2] = entity_number[tail_mask].to(self.device)
        return batch_neg


class BernNegativeSampler():
    def __init__(self, triples, entity_dict_len, relation_dict_len, device=torch.device('cuda:0')):
        # tensor(len,3)
        self.triples = triples
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.device = device

        h_r_uniq, t_count = torch.unique(triples[:, :-1], return_counts=True, dim=0)
        r_t_uniq, h_count = torch.unique(triples[:, 1:], return_counts=True, dim=0)

        self.P_remove_head = torch.zeros(self.relation_dict_len)
        for r in range(self.relation_dict_len):
            idx = h_r_uniq[:, 1] == r
            tph = torch.mean(t_count[idx].type(torch.FloatTensor))

            idx = r_t_uniq[:, 0] == r
            hpt = torch.mean(h_count[idx].type(torch.FloatTensor))

            self.P_remove_head[r] = tph / (tph + hpt)

    def create_negative(self, batch_pos):
        # tensor(batch,3)
        batch_neg = batch_pos.clone()
        entity_number = torch.randint(self.entity_dict_len, (batch_neg.size()[0],)).to(self.device)
        relation = batch_pos[:, 1]
        mask = torch.rand(batch_neg.size()[0])
        head_mask = (mask < self.P_remove_head[relation]).to(self.device)
        tail_mask = (mask >= self.P_remove_head[relation]).to(self.device)
        batch_neg[head_mask, 0] = entity_number[head_mask].to(self.device)
        batch_neg[tail_mask, 2] = entity_number[tail_mask].to(self.device)
        return batch_neg


class AdversarialSampler:
    def __init__(self, triples, entity_dict_len, relation_dict_len, neg_per_pos, device=torch.device('cuda:0')):
        # tensor(len,3)
        self.triples = triples
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.neg_per_pos = neg_per_pos
        self.device = device

    def create_negative(self, batch_pos):
        """
        batch_pos:(batch,3)
        return: batch_neg(batch * neg_per_pos,3)
        """
        return torch.cat([self._create_negative(batch_pos) for i in range(self.neg_per_pos)], dim=0)

    def _create_negative(self, batch_pos):
        batch_neg = batch_pos.clone()
        entity_number = torch.randint(self.entity_dict_len, (batch_neg.size()[0],)).to(self.device)
        mask = torch.rand(batch_neg.size()[0])
        head_mask = (mask > 0.5).bool().to(self.device)
        tail_mask = (mask <= 0.5).bool().to(self.device)
        batch_neg[head_mask, 0] = entity_number[head_mask].to(self.device)
        batch_neg[tail_mask, 2] = entity_number[tail_mask].to(self.device)
        return batch_neg


if __name__ == "__main__":
    fake_triples = torch.tensor([[1, 0, 0],
                                 [1, 0, 4],
                                 [1, 0, 3],
                                 [5, 0, 3],
                                 [2, 1, 5],
                                 [4, 2, 2]]).to("cuda:0")
    batch_pos = torch.tensor([[4, 2, 1],
                              [0, 3, 4]]).to("cuda:0")
    # sampler=UnifNegativeSampler(fake_triples,5,4)
    # batch_neg=sampler.create_negative(batch_pos)
    sampler = BernNegativeSampler(fake_triples, 6, 3)
    batch_neg = sampler.create_negative(batch_pos)
    # sampler=AdversarialSampler(fake_triples,5,4,3)
    # batch_neg=sampler.create_negative(batch_pos)
    print(batch_neg)

########################################################################################################################
# import random
#
# import numpy as np
# import torch
#
#
# class UnifNegativeSampler():
#     def __init__(self, triples, entity_dict_len, relation_dict_len, device=torch.device('cuda:0')):
#         # (batch,3)
#         self.triples = triples
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.device = device
#
#     def create_negative(self, batch_pos):
#         batch_neg = batch_pos.clone()
#         entity_number = torch.randint(self.entity_dict_len, (batch_neg.size()[0],)).to(self.device)
#         mask = torch.rand(batch_neg.size()[0])
#         head_mask = (mask > 0.5).bool().to(self.device)
#         tail_mask = (mask <= 0.5).bool().to(self.device)
#         batch_neg[head_mask, 0] = entity_number[head_mask].to(self.device)
#         batch_neg[tail_mask, 2] = entity_number[tail_mask].to(self.device)
#         return batch_neg
#
#
# class BernNegativeSampler():
#     # TODO: The for loop need to be removed!
#     def __init__(self, triples, entity_dict_len, relation_dict_len):
#         # numpy:(batch,3)
#         self.triples = triples
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#
#         h_r_uniq, t_count = np.unique(triples[:, :-1], return_counts=True, axis=0)
#         r_t_uniq, h_count = np.unique(triples[:, 1:], return_counts=True, axis=0)
#
#         self.P_remove_head = np.zeros(self.relation_dict_len)
#         for r in range(self.relation_dict_len):
#             idx = h_r_uniq[:, 1] == r
#             tph = np.mean(t_count[idx])
#
#             idx = r_t_uniq[:, 0] == r
#             hpt = np.mean(h_count[idx])
#
#             self.P_remove_head[r] = tph / (tph + hpt)
#
#     def create_negative(self, batch_pos):
#         # batch_pos:tensr (batch,3)
#         batch_neg = batch_pos.clone().detach()
#         for i in range(len(batch_pos)):
#             relation = int(batch_pos[i][1].item())
#             if (random.random() < self.P_remove_head[relation]):
#                 # corrupt head
#                 batch_neg[i][0] = np.random.randint(0, self.entity_dict_len)
#             else:
#                 # corrupt tail
#                 batch_neg[i][2] = np.random.randint(0, self.entity_dict_len)
#
#         return batch_neg
#
#
# class AdversarialSampler:
#     # TODO: The for loop need to be removed!
#     def __init__(self, triples, entity_dict_len, relation_dict_len, neg_per_pos):
#         # (batch,3)
#         self.triples = triples
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.neg_per_pos = neg_per_pos
#
#     def create_negative(self, batch_pos):
#         """
#         batch_pos:(batch,3)
#         return: batch_neg(batch * neg_per_pos,3)
#         """
#         return torch.cat([self._create_negative(batch_pos) for i in range(self.neg_per_pos)], dim=0)
#
#     def _create_negative(self, batch_pos):
#         batch_neg = batch_pos.clone().detach()
#         for i in range(len(batch_pos)):
#             if (random.random() < 0.5):
#                 # corrupt head
#                 batch_neg[i][0] = np.random.randint(0, self.entity_dict_len)
#             else:
#                 # corrupt tail
#                 batch_neg[i][2] = np.random.randint(0, self.entity_dict_len)
#
#         return batch_neg
#