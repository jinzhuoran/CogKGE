import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from transformers import RobertaModel

####################TransE_baseline###############################
class TransE_baseline(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,p=1.0):
        super(TransE_baseline, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.name = "TransE_baseline"
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
        head_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)

        head_embedding = F.normalize(head_embedding, p=2, dim=2)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=2)

        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)

####################TransE_Add_Description###############################

class TransE_Add_Description(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,node_lut,p=1.0):
        super(TransE_Add_Description, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.node_lut=node_lut
        self.token=node_lut.token
        self.mask=node_lut.mask
        self.name = "TransE_Add_Description"
        self.square = embedding_dim ** 0.5
        self.p = p
        self.pre_training_model_name = "roberta-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pre_training_model_name)
        self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)
        self.out_dim=self.pre_training_model.pooler.dense.out_features
        self.liner=nn.Linear(self.out_dim, embedding_dim)
        # self.transfer_matrix=nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim*768)

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
        current_device="cuda:%s"%(torch.cuda.current_device())
        triplet_idx=triplet_idx[:,:3]
        head_token=self.token[triplet_idx[:,0]].to(current_device)
        tail_token=self.token[triplet_idx[:,2]].to(current_device)
        head_mask=self.mask[triplet_idx[:,0]].to(current_device)
        tail_mask=self.mask[triplet_idx[:,2]].to(current_device)
        head_pertrain_embedding=self.pre_training_model(head_token,attention_mask=head_mask).pooler_output
        tail_pertrain_embedding=self.pre_training_model(tail_token,attention_mask=tail_mask).pooler_output
        head_embedding=self.liner(head_pertrain_embedding)
        tail_embedding=self.liner(tail_pertrain_embedding)
        # head_embedding=self.transfer_matrix.view(-1, self.dim_entity, self.dim_relation)
        # tail_embedding=self.transfer_matrix.view(-1, self.dim_entity, self.dim_relation)

        head_embedding = torch.unsqueeze(head_embedding, 1)
        relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embedding = torch.unsqueeze(tail_embedding, 1)

        head_embedding = F.normalize(head_embedding, p=2, dim=2)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=2)

        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)



####################TransE_Add_Time###############################
class TransE_Add_Time(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,time_lut,p=1.0):
        super(TransE_Add_Time, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.name = "TransE_Add_Time"
        self.square = embedding_dim ** 0.5
        self.p = p
        self.time_lut=time_lut
        self.time_transfer_matrix=nn.Embedding(num_embeddings=len(self.time_lut), embedding_dim=self.embedding_dim*self.embedding_dim)

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
        head_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)
        start=triplet_idx[:, 3]
        end=triplet_idx[:, 4]

        head_embedding=torch.bmm(head_embedding,self.time_transfer_matrix(start).view(-1,self.embedding_dim,self.embedding_dim)+
                                 self.time_transfer_matrix(end).view(-1,self.embedding_dim,self.embedding_dim))
        tail_embedding=torch.bmm(tail_embedding,self.time_transfer_matrix(start).view(-1,self.embedding_dim,self.embedding_dim)+
                                 self.time_transfer_matrix(end).view(-1,self.embedding_dim,self.embedding_dim))

        head_embedding = F.normalize(head_embedding, p=2, dim=2)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=2)

        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)


####################TransE_Add_Type###############################
class TransE_Add_Type(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,node_lut,p=1.0):
        super(TransE_Add_Type, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.name = "TransE_Add_Type"
        self.square = embedding_dim ** 0.5
        self.p = p
        self.node_lut=node_lut
        self.type=node_lut.type
        self.type_len=max(node_lut.type)+1
        self.type_transfer_matrix=nn.Embedding(num_embeddings=self.type_len, embedding_dim=self.embedding_dim*self.embedding_dim)

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
        current_device="cuda:%s"%(torch.cuda.current_device())
        triplet_idx=triplet_idx[:,:3]
        head_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)

        head_type=self.type[triplet_idx[:, 0]].to(current_device)
        tail_type=self.type[triplet_idx[:, 2]].to(current_device)

        head_embedding=torch.bmm(head_embedding,self.type_transfer_matrix(head_type).view(-1,self.embedding_dim,self.embedding_dim))
        tail_embedding=torch.bmm(tail_embedding,self.type_transfer_matrix(tail_type).view(-1,self.embedding_dim,self.embedding_dim))

        head_embedding = F.normalize(head_embedding, p=2, dim=2)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=2)

        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)



####################TransE_Add_Path###############################
class TransE_Add_Path(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, embedding_dim,p=1.0):
        super(TransE_Add_Path, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.embedding_dim = embedding_dim
        self.name = "TransE_Add_Path"
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
        head_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
        relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
        tail_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)

        head_embedding = F.normalize(head_embedding, p=2, dim=2)
        tail_embedding = F.normalize(tail_embedding, p=2, dim=2)

        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
        return score  # (batch,)
############################################################OLDOLD######################################################
############################################################OLDOLD######################################################
####################TTD_TransR_TYPE_3_Baseline_bian###############################
####################type(×)，time（x）,description(x)##############################
# class TTD_TransR_TYPE_3_Baseline_bian(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, dim_entity, dim_relation, p=2.0):
#         super(TTD_TransR_TYPE_3_Baseline_bian, self).__init__()
#         self.name = "TTD_TransR_TYPE_3_Baseline_bian"
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.dim_entity = dim_entity
#         self.dim_relation = dim_relation
#         self.p = p
#
#         self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
#         self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
#         self.h_transfer_matrix = nn.Embedding(relation_dict_len , dim_entity * dim_relation)
#         self.t_transfer_matrix = nn.Embedding(relation_dict_len , dim_entity * dim_relation)
#
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#         nn.init.xavier_uniform_(self.h_transfer_matrix.weight.data)
#         nn.init.xavier_uniform_(self.t_transfer_matrix.weight.data)
#
#     def transfer(self, e, r_transfer):
#         r_transfer = r_transfer.view(-1, self.dim_entity, self.dim_relation)
#         e = torch.unsqueeze(e, 1)
#
#         return torch.squeeze(torch.bmm(e, r_transfer))
#
#     def get_score(self, sample):
#         output = self._forward(sample)
#         score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
#         return score  # (batch,)
#
#     def get_embedding(self, sample):
#         return self._forward(sample)
#
#     def forward(self, sample):
#         return self.get_score(sample)
#
#     def _forward(self, sample):  # sample:(batch,3)
#         batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
#         h = self.entity_embedding(batch_h)
#         r = self.relation_embedding(batch_r)
#         t = self.entity_embedding(batch_t)
#
#         h_r_transfer = self.h_transfer_matrix(batch_r)
#         t_r_transfer = self.t_transfer_matrix(batch_r)
#
#         h = F.normalize(h, p=2.0, dim=-1)
#         t = F.normalize(t, p=2.0, dim=-1)  # ||h|| <= 1  ||t|| <= 1
#
#         h = self.transfer(h, h_r_transfer)
#         t = self.transfer(t, t_r_transfer)
#
#         h = F.normalize(h, p=2.0, dim=-1)
#         r = F.normalize(r, p=2.0, dim=-1)
#         t = F.normalize(t, p=2.0, dim=-1)
#
#         h = torch.unsqueeze(h, 1)
#         r = torch.unsqueeze(r, 1)
#         t = torch.unsqueeze(t, 1)
#
#         return torch.cat((h, r, t), dim=1)
#
# ####################TTD_TransE_TYPE_1######################################
# ####################type(√)，time（x）,description(x)#######################
# class TTD_TransE_TYPE(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, node_lut,embedding_dim,p=1.0):
#         super(TTD_TransE_TYPE, self).__init__()
#         self.name = "TTD_TransE_TYPE"
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.embedding_dim = int(embedding_dim/2)
#         self.square = embedding_dim ** 0.5
#         self.p = p
#         self.node_lut=node_lut
#         self.node_type=node_lut.type
#         self.type_dict_len=max(self.node_type).item()+1
#
#         self.type_embedding = nn.Embedding(num_embeddings=self.type_dict_len, embedding_dim=self.embedding_dim)
#         self.type_embedding.weight.data = torch.FloatTensor(self.type_dict_len, self.embedding_dim).uniform_(
#             -6 / self.square, 6 / self.square)
#         self.type_embedding.weight.data = F.normalize(self.type_embedding.weight.data, p=2, dim=1)
#
#         self.entity_embedding = nn.Embedding(num_embeddings=self.entity_dict_len, embedding_dim=self.embedding_dim)
#         self.entity_embedding.weight.data = torch.FloatTensor(self.entity_dict_len, self.embedding_dim).uniform_(
#             -6 / self.square, 6 / self.square)
#         self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
#         self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim*2)
#         self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim*2).uniform_(
#             -6 / self.square, 6 / self.square)
#         self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)
#         self.test_memory=torch.ones(6,6)
#
#
#     def forward(self,sample):
#         current_device="cuda:%s"%(torch.cuda.current_device())
#         triplet_idx=sample[:,:3]
#         head_instance_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 0]), 1)
#         relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)
#         tail_instance_embedding = torch.unsqueeze(self.entity_embedding(triplet_idx[:, 2]), 1)
#         head_type_embedding=torch.unsqueeze(self.type_embedding(self.node_type[triplet_idx[:, 0]].to(current_device)), 1)
#         tail_type_embedding=torch.unsqueeze(self.type_embedding(self.node_type[triplet_idx[:, 2]].to(current_device)), 1)
#         head_embedding=torch.cat([head_instance_embedding,head_type_embedding],dim=2)
#         tail_embedding=torch.cat([tail_instance_embedding,tail_type_embedding],dim=2)
#         head_embedding = F.normalize(head_embedding, p=2, dim=2)
#         tail_embedding = F.normalize(tail_embedding, p=2, dim=2)
#
#         triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)
#
#         output = triplet_embedding # (batch,3,embedding_dim)
#         score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
#         return score  # (batch,)
#
#
#
# ####################TTD_TransR_TYPE_3######################################
# ####################type(√)，time（x）,description(x)#######################
#
# class TTD_TransR_TYPE_3(nn.Module):
#     def __init__(self, entity_dict_len, relation_dict_len, dim_entity, dim_relation, node_lut, p=2.0):
#         super(TTD_TransR_TYPE_3, self).__init__()
#         self.name = "TTD_TransR_Type_3"
#         self.entity_dict_len = entity_dict_len
#         self.relation_dict_len = relation_dict_len
#         self.dim_entity = dim_entity
#         self.dim_relation = dim_relation
#         self.p = p
#         self.node_lut=node_lut
#         self.node_type=node_lut.type
#         self.type_dict_len = max(self.node_type).item() + 1
#
#         self.entity_embedding = nn.Embedding(entity_dict_len, dim_entity)
#         self.relation_embedding = nn.Embedding(relation_dict_len, dim_relation)
#         self.h_transfer_matrix = nn.Embedding(relation_dict_len * self.type_dict_len, dim_entity * dim_relation)
#         self.t_transfer_matrix = nn.Embedding(relation_dict_len * self.type_dict_len, dim_entity * dim_relation)
#
#         nn.init.xavier_uniform_(self.entity_embedding.weight.data)
#         nn.init.xavier_uniform_(self.relation_embedding.weight.data)
#         nn.init.xavier_uniform_(self.h_transfer_matrix.weight.data)
#         nn.init.xavier_uniform_(self.t_transfer_matrix.weight.data)
#
#     def transfer(self, e, r_transfer):
#         r_transfer = r_transfer.view(-1, self.dim_entity, self.dim_relation)
#         e = torch.unsqueeze(e, 1)
#
#         return torch.squeeze(torch.bmm(e, r_transfer))
#
#     def get_score(self, sample):
#         output = self._forward(sample)
#         score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=self.p)
#         return score  # (batch,)
#
#     def get_embedding(self, sample):
#         return self._forward(sample)
#
#     def forward(self, sample):
#         return self.get_score(sample)
#
#     def _forward(self, sample):  # sample:(batch,3)
#         current_device = "cuda:%s" % (torch.cuda.current_device())
#         batch_h, batch_r, batch_t = sample[:, 0], sample[:, 1], sample[:, 2]
#         h = self.entity_embedding(batch_h)
#         r = self.relation_embedding(batch_r)
#         t = self.entity_embedding(batch_t)
#
#         h_r_transfer = self.h_transfer_matrix(batch_r*self.type_dict_len+self.node_type[batch_h].to(current_device))
#         t_r_transfer = self.t_transfer_matrix(batch_r * self.type_dict_len + self.node_type[batch_t].to(current_device))
#
#         h = F.normalize(h, p=2.0, dim=-1)
#         t = F.normalize(t, p=2.0, dim=-1)  # ||h|| <= 1  ||t|| <= 1
#
#         h = self.transfer(h, h_r_transfer)
#         t = self.transfer(t, t_r_transfer)
#
#         h = F.normalize(h, p=2.0, dim=-1)
#         r = F.normalize(r, p=2.0, dim=-1)
#         t = F.normalize(t, p=2.0, dim=-1)
#
#         h = torch.unsqueeze(h, 1)
#         r = torch.unsqueeze(r, 1)
#         t = torch.unsqueeze(t, 1)
#
#         return torch.cat((h, r, t), dim=1)








