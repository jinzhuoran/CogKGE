import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaModel
from transformers import RobertaTokenizer


class KEPLER(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, node_lut=None):
        super(KEPLER, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.node_lut=node_lut
        self.name = "KEPLER"
        self.pre_training_model_name = "roberta-base"
        self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)
        self.out_dim=self.pre_training_model.pooler.dense.out_features
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.out_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def get_score(self,triplet_idx):
        output = self._forward(triplet_idx)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
        return score

    def forward(self,triplet_idx):
        return self.get_score(triplet_idx)

    def get_embedding(self,triplet_idx):
        current_device="cuda:%s"%(torch.cuda.current_device())
        head_token=self.node_lut.token[triplet_idx[:,0]].to(current_device)
        tail_token=self.node_lut.token[triplet_idx[:,2]].to(current_device)
        head_mask=self.node_lut.mask[triplet_idx[:,0]].to(current_device)
        tail_mask=self.node_lut.mask[triplet_idx[:,2]].to(current_device)
        head_embedding=self.pre_training_model(head_token,attention_mask=head_mask).pooler_output
        tail_embedding=self.pre_training_model(tail_token,attention_mask=tail_mask).pooler_output
        relation_embedding = self.relation_embedding(triplet_idx[:, 1])
        return head_embedding,relation_embedding,tail_embedding

    def _forward(self, triplet_idx):
        current_device="cuda:%s"%(torch.cuda.current_device())
        head_embedding,relation_embedding,tail_embedding=self.get_embedding(triplet_idx)


        head_embedding = torch.unsqueeze(head_embedding , 1)
        relation_embedding = torch.unsqueeze(relation_embedding , 1)
        tail_embedding = torch.unsqueeze(tail_embedding, 1)
        triplet_embedding = torch.cat([head_embedding, relation_embedding, tail_embedding], dim=1)

        output = triplet_embedding

        return output

    # def get_score(self, triplet_idx):
    #     output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
    #     score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
    #     return score  # (batch,)
    #
    # def forward(self, triplet_idx):
    #     return self.get_score(triplet_idx)
    #
    # def get_embedding(self, triplet_idx):
    #     return self._forward(triplet_idx)
    #
    # def _forward(self, triplet_idx):
    #     current_device = "cuda:%s" % (torch.cuda.current_device())
    #     head_tokens_tensor = self.tokens(triplet_idx[:, 0]).type(torch.LongTensor).to(current_device)
    #     head_tokens_tensor.requires_grad_(False)
    #     tail_tokens_tensor = self.tokens(triplet_idx[:, 2]).type(torch.LongTensor).to(current_device)
    #     tail_tokens_tensor.requires_grad_(False)
    #     head_masks_tensor = self.masks(triplet_idx[:, 0]).type(torch.LongTensor).to(current_device)
    #     head_masks_tensor.requires_grad_(False)
    #     tail_masks_tensor = self.masks(triplet_idx[:, 2]).type(torch.LongTensor).to(current_device)
    #     tail_masks_tensor.requires_grad_(False)
    #     head_roberta = self.pre_training_model(head_tokens_tensor.to(current_device),
    #                                            token_type_ids=None,
    #                                            attention_mask=head_masks_tensor)  # head_roberta.pooler_output.shape  （batch_size,768）
    #     tail_roberta = self.pre_training_model(tail_tokens_tensor.to(current_device),
    #                                            token_type_ids=None,
    #                                            attention_mask=tail_masks_tensor)
    #     # entity的embedding
    #     head_embedding = torch.unsqueeze(head_roberta.pooler_output, dim=1)  # (batch_size,1,768)oxiangxia
    #     tail_embedding = torch.unsqueeze(tail_roberta.pooler_output, dim=1)  # (batch_size,1,768)
    #     # relation的embedding表示
    #     relation_embedding = torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)  # (batch_size,1,768)
    #     # 输出向量拼接
    #     triplet_embedding = torch.cat([head_embedding, tail_embedding, relation_embedding], dim=1)
    #     output = triplet_embedding  # (batch,3,embedding_dim)
    #     return output
