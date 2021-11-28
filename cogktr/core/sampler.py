import random
import numpy as np
import torch

class UnifNegativeSampler():
    def __init__(self,triples,entity_dict_len,relation_dict_len):
        # (batch,3)
        self.triples = triples
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
    
    def create_negative(self,batch_pos):
        batch_neg = batch_pos.clone().detach()
        for i in range(len(batch_pos)):
            if(random.random() < 0.5):
                # corrupt head
                batch_neg[i][0] = np.random.randint(0,self.entity_dict_len)
            else:
                # corrupt tail
                batch_neg[i][2] = np.random.randint(0,self.entity_dict_len)

        return batch_neg

class BernNegativeSampler():
    def __init__(self,triples,entity_dict_len,relation_dict_len):
        # numpy:(batch,3)
        self.triples = triples
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len

        h_r_uniq, t_count = np.unique(triples[:,:-1], return_counts=True,axis=0)
        r_t_uniq, h_count = np.unique(triples[:,1:],return_counts=True,axis=0)

        self.P_remove_head = np.zeros(self.relation_dict_len)
        for r in range(self.relation_dict_len):
            idx = h_r_uniq[:,1] == r
            tph = np.mean(t_count[idx])

            idx = r_t_uniq[:,0] == r
            hpt = np.mean(h_count[idx])

            self.P_remove_head[r] = tph/(tph+hpt)
        
    def create_negative(self,batch_pos):
        # batch_pos:tensr (batch,3)
        batch_neg = batch_pos.clone().detach()
        for i in range(len(batch_pos)):
            relation = int(batch_pos[i][1].item())
            if(random.random() < self.P_remove_head[relation]):
                # corrupt head
                batch_neg[i][0] = np.random.randint(0,self.entity_dict_len)
            else:
                # corrupt tail
                batch_neg[i][2] = np.random.randint(0,self.entity_dict_len)

        return batch_neg

class AdversarialSampler:
    def __init__(self,triples,entity_dict_len,relation_dict_len,neg_per_pos):
        # (batch,3)
        self.triples = triples
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.neg_per_pos = neg_per_pos

    def create_negative(self,batch_pos):
        """
        batch_pos:(batch,3)
        return: batch_neg(batch * neg_per_pos,3)
        """
        return torch.cat([self._create_negative(batch_pos) for i in range(self.neg_per_pos)],dim=0)
        

    def _create_negative(self,batch_pos):
        batch_neg = batch_pos.clone().detach()
        for i in range(len(batch_pos)):
            if(random.random() < 0.5):
                # corrupt head
                batch_neg[i][0] = np.random.randint(0,self.entity_dict_len)
            else:
                # corrupt tail
                batch_neg[i][2] = np.random.randint(0,self.entity_dict_len)

        return batch_neg    

    




