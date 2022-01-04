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

        self.type_embedding = nn.Embedding(num_embeddings=self.type_dict_len*self.relation_dict_len, embedding_dim=(self.embedding_dim*self.embedding_dim))
        self.type_embedding.weight.data = torch.FloatTensor(self.type_dict_len*self.relation_dict_len, self.embedding_dim*self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.type_embedding.weight.data = F.normalize(self.type_embedding.weight.data, p=2, dim=1)

        self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
        self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)
        self.test_memory=torch.ones(6,6)


    def forward(self,sample):
        current_device="cuda:%s"%(torch.cuda.current_device())
        triplet_idx=sample[:,:3]
        head_instance_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 2)
        relation_embeddiing = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 2)
        tail_instance_embeddiing = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 2)
        head_index=triplet_idx[:, 1]*(self.type_dict_len)+self.node_type[triplet_idx[:, 0]].to(current_device)
        tail_index=triplet_idx[:, 1]*(self.type_dict_len)+self.node_type[triplet_idx[:, 2]].to(current_device)
        head_embeddiing=torch.bmm(self.type_embedding(head_index).view(-1,self.embedding_dim,self.embedding_dim),head_instance_embeddiing)
        tail_embeddiing=torch.bmm(self.type_embedding(tail_index).view(-1,self.embedding_dim,self.embedding_dim),tail_instance_embeddiing)

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





