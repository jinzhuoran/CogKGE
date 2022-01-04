import torch
import torch.nn as nn
import torch.nn.functional as F

####################TTD_TransE_baseline###############################
####################type(×)，time（x）,description(x)##################
class TTD_TransE(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,p=1.0):
        super(TTD_TransE, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.name = "TTD_TransE"
        self.square = embedding_dim ** 0.5
        self.p = p

        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)
        self.test_memory=torch.ones(6,6)

    def forward(self,triplet_idx):
        triplet_idx=triplet_idx[:,:3]
        head_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embeddiing = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)

        head_embeddiing = F.normalize(head_embeddiing, p=2, dim=2)
        tail_embeddiing = F.normalize(tail_embeddiing, p=2, dim=2)

        triplet_embedding = torch.cat([head_embeddiing, relation_embeddiing, tail_embeddiing], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)


####################TTD_TransR_TYPE_3_Baseline_bian###############################
####################type(×)，time（x）,description(x)##############################
class TTD_TransR_TYPE_3_Baseline_bian(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, dim_entity, dim_relation, p=2.0):
        super(TTD_TransR_TYPE_3_Baseline_bian, self).__init__()
        self.name = "TTD_TransR_TYPE_3_Baseline_bian"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.dim_entity = dim_entity
        self.dim_relation = dim_relation
        self.p = p

        self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
        self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
        self.h_transfer_matrix = nn.Embedding(relation_dict_len , dim_entity * dim_relation)
        self.t_transfer_matrix = nn.Embedding(relation_dict_len , dim_entity * dim_relation)

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.h_transfer_matrix.weight.data)
        nn.init.xavier_uniform_(self.t_transfer_matrix.weight.data)

    def transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.dim_entity, self.dim_relation)
        e = torch.unsqueeze(e, 1)

        return torch.squeeze(torch.bmm(e, r_transfer))

    def get_score(self, sample):
        output = self._forward(sample)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)

    def get_embedding(self, sample):
        return self._forward(sample)

    def forward(self, sample):
        return self.get_score(sample)

    def _forward(self, sample):  # sample:(batch,3)
        batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
        h = self.entity_embedding(batch_h)
        r = self.relation_embedding(batch_r)
        t = self.entity_embedding(batch_t)

        h_r_transfer = self.h_transfer_matrix(batch_r)
        t_r_transfer = self.t_transfer_matrix(batch_r)

        h = F.normalize(h, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)  # ||h|| <= 1  ||t|| <= 1

        h = self.transfer(h, h_r_transfer)
        t = self.transfer(t, t_r_transfer)

        h = F.normalize(h, p=2.0, dim=-1)
        r = F.normalize(r, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)

        h = torch.unsqueeze(h, 1)
        r = torch.unsqueeze(r, 1)
        t = torch.unsqueeze(t, 1)

        return torch.cat((h, r, t), dim=1)

####################TTD_TransE_TYPE_1######################################
####################type(√)，time（x）,description(x)#######################
class TTD_TransE_TYPE(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, node_lut,embedding_dim,p=1.0):
        super(TTD_TransE_TYPE, self).__init__()
        self.name = "TTD_TransE_TYPE"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = int(embedding_dim/2)
        self.square = embedding_dim ** 0.5
        self.p = p
        self.node_lut=node_lut
        self.node_type=node_lut.type
        self.type_dict_len=max(self.node_type).item()+1

        self.type_embedding = nn.Embedding(num_embeddings=self.type_dict_len, embedding_dim=self.embedding_dim)
        self.type_embedding.weight.data = torch.FloatTensor(self.type_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.type_embedding.weight.data = F.normalize(self.type_embedding.weight.data, p=2, dim=1)

        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim*2)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim*2).uniform_(
            -6 / self.square, 6 / self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)
        self.test_memory=torch.ones(6,6)


    def forward(self,sample):
        current_device="cuda:%s"%(torch.cuda.current_device())
        triplet_idx=sample[:,:3]
        head_instance_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embeddiing = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_instance_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)
        head_type_embedding=torch.unsqueeze(self.type_embedding(self.node_type[triplet_idx[:, 0]].to(current_device)), 1)
        tail_type_embedding=torch.unsqueeze(self.type_embedding(self.node_type[triplet_idx[:, 2]].to(current_device)), 1)
        head_embeddiing=torch.cat([head_instance_embeddiing,head_type_embedding],dim=2)
        tail_embeddiing=torch.cat([tail_instance_embeddiing,tail_type_embedding],dim=2)
        head_embeddiing = F.normalize(head_embeddiing, p=2, dim=2)
        tail_embeddiing = F.normalize(tail_embeddiing, p=2, dim=2)

        triplet_embedding = torch.cat([head_embeddiing, relation_embeddiing, tail_embeddiing], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)



####################TTD_TransE_TYPE_2######################################
####################type(√)，time（x）,description(x)#######################
class TTD_TransE_TYPE_2(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, node_lut,embedding_dim,p=1.0):
        super(TTD_TransE_TYPE_2, self).__init__()
        self.name = "TTD_TransE_TYPE"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.square = embedding_dim ** 0.5
        self.p = p
        self.node_lut=node_lut
        self.node_type=node_lut.type
        self.type_dict_len=max(self.node_type).item()+1

        # self.type_matrix=torch.rand(2,self.relation_dict_len,self.type_dict_len,self.embedding_dim,self.embedding_dim)
        # self.type_matrix=F.normalize(self.type_matrix, p=2, dim=3)
        #[2, 389, 822, 50, 50]

        # self.head_type_embedding = nn.Embedding(num_embeddings=self.type_dict_len*self.relation_dict_len,
        #                                         embedding_dim=(self.embedding_dim*self.embedding_dim))
        #
        # self.tail_type_embedding = nn.Embedding(num_embeddings=self.type_dict_len * self.relation_dict_len,
        #                                         embedding_dim=(self.embedding_dim * self.embedding_dim))

        self.head_type_embedding = nn.Embedding(num_embeddings=self.relation_dict_len,
                                                embedding_dim=(self.embedding_dim*self.embedding_dim))

        self.tail_type_embedding = nn.Embedding(num_embeddings=self.relation_dict_len,
                                                embedding_dim=(self.embedding_dim * self.embedding_dim))




        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        # self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
        #     -6 / self.square, 6 / self.square)
        # self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        # self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
        #     -6 / self.square, 6 / self.square)
        # self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.head_type_embedding.weight.data)
        nn.init.xavier_uniform_(self.tail_type_embedding.weight.data)


    def forward(self,sample):
        current_device="cuda:%s"%(torch.cuda.current_device())
        triplet_idx=sample[:,:3]
        head_instance_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 2)
        relation_embeddiing = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 2)
        tail_instance_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 2)
        # head_index=triplet_idx[:, 1]*(self.type_dict_len)+self.node_type[triplet_idx[:, 0]].to(current_device)
        # tail_index=triplet_idx[:, 1]*(self.type_dict_len)+self.node_type[triplet_idx[:, 2]].to(current_device)
        head_index=triplet_idx[:, 1]
        tail_index=triplet_idx[:, 1]
        head_embeddiing=torch.bmm(self.head_type_embedding(head_index).view(-1,self.embedding_dim,self.embedding_dim),head_instance_embeddiing)
        tail_embeddiing=torch.bmm(self.tail_type_embedding(tail_index).view(-1,self.embedding_dim,self.embedding_dim),tail_instance_embeddiing)

        # head_embeddiing=torch.bmm(self.type_matrix[0,triplet_idx[:, 1],self.node_type[triplet_idx[:, 0]]].to(current_device),
        #                          head_instance_embeddiing)
        # tail_embeddiing=torch.bmm(self.type_matrix[1,triplet_idx[:, 1],self.node_type[triplet_idx[:, 2]]].to(current_device),
        #                          tail_instance_embeddiing)

        # self.type_matrix=F.normalize(self.type_matrix, p=2, dim=3)
        head_embeddiing = F.normalize(head_embeddiing, p=2, dim=2)
        tail_embeddiing = F.normalize(tail_embeddiing, p=2, dim=2)

        triplet_embedding = torch.cat([head_embeddiing, relation_embeddiing, tail_embeddiing], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)


####################TTD_TransR_TYPE_3######################################
####################type(√)，time（x）,description(x)#######################

class TTD_TransR_TYPE_3(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, dim_entity, dim_relation, node_lut, p=2.0):
        super(TTD_TransR_TYPE_3, self).__init__()
        self.name = "TTD_TransR_Type_3"
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.dim_entity = dim_entity
        self.dim_relation = dim_relation
        self.p = p
        self.node_lut=node_lut
        self.node_type=node_lut.type
        self.type_dict_len = max(self.node_type).item() + 1

        self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
        self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
        self.h_transfer_matrix = nn.Embedding(relation_dict_len * self.type_dict_len, dim_entity * dim_relation)
        self.t_transfer_matrix = nn.Embedding(relation_dict_len * self.type_dict_len, dim_entity * dim_relation)

        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.h_transfer_matrix.weight.data)
        nn.init.xavier_uniform_(self.t_transfer_matrix.weight.data)

    def transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.dim_entity, self.dim_relation)
        e = torch.unsqueeze(e, 1)

        return torch.squeeze(torch.bmm(e, r_transfer))

    def get_score(self, sample):
        output = self._forward(sample)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)

    def get_embedding(self, sample):
        return self._forward(sample)

    def forward(self, sample):
        return self.get_score(sample)

    def _forward(self, sample):  # sample:(batch,3)
        current_device = "cuda:%s" % (torch.cuda.current_device())
        batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
        h = self.entity_embedding(batch_h)
        r = self.relation_embedding(batch_r)
        t = self.entity_embedding(batch_t)

        h_r_transfer = self.h_transfer_matrix(batch_r*self.type_dict_len+self.node_type[batch_h].to(current_device))
        t_r_transfer = self.t_transfer_matrix(batch_r * self.type_dict_len + self.node_type[batch_t].to(current_device))

        h = F.normalize(h, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)  # ||h|| <= 1  ||t|| <= 1

        h = self.transfer(h, h_r_transfer)
        t = self.transfer(t, t_r_transfer)

        h = F.normalize(h, p=2.0, dim=-1)
        r = F.normalize(r, p=2.0, dim=-1)
        t = F.normalize(t, p=2.0, dim=-1)

        h = torch.unsqueeze(h, 1)
        r = torch.unsqueeze(r, 1)
        t = torch.unsqueeze(t, 1)

        return torch.cat((h, r, t), dim=1)








