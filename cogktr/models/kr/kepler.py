import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from transformers import RobertaModel
from tqdm import tqdm


class KEPLER(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, token_length,embedding_dim):
        super(KEPLER, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.token_length=token_length
        self.embedding_dim = embedding_dim
        self.name = "KEPLER"
        self.lookuptable_E=None
        self.lookuptable_R=None
        self.pre_training_model_name = "roberta-base"
        self.tokens = None
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pre_training_model_name)
        self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)
        self.square = embedding_dim ** 0.5
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)

    def load_lookuotable(self,lookuptable_E,lookuptable_R):
        self.lookuptable_E = lookuptable_E
        self.lookuptable_R = lookuptable_R
        # 在lookuptable生成token的字典
        entity_descriptions_dict=dict()
        print("Descriptions Tokenization ... ")
        tokens_list=list()
        for i in tqdm(range(len(self.lookuptable_E))):
            encoded_text = self.tokenizer.encode(
                self.lookuptable_E["descriptions"][i],
                add_special_tokens=True,
                max_length=self.token_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            entity_descriptions_dict[self.lookuptable_E["name"][i]]=encoded_text
            tokens_list.append(encoded_text)
        self.lookuptable_E("tokens",entity_descriptions_dict)
        self.tokens = nn.Embedding(num_embeddings=len(self.lookuptable_E), embedding_dim=self.token_length)
        self.tokens.weight.data = torch.cat(tokens_list, dim=0).type(torch.FloatTensor)
        self.lookuptable_E.print_table(2)

    def get_score(self,triplet_idx):
        output = self._forward(triplet_idx)  # (batch,3,embedding_dim)
        score = F.pairwise_distance(output[:, 0] + output[:, 1], output[:, 2], p=2)
        return score  # (batch,)

    def forward(self,triplet_idx):
        return self.get_score(triplet_idx)


    def get_embedding(self,triplet_idx):
        return self._forward(triplet_idx)


    def _forward(self, triplet_idx):
        current_device="cuda:%s"%(torch.cuda.current_device())
        head_tokens_tensor=self.tokens(triplet_idx[:, 0]).type(torch.LongTensor).to(current_device)
        head_tokens_tensor.requires_grad_(False)
        tail_tokens_tensor= self.tokens(triplet_idx[:, 2]).type(torch.LongTensor).to(current_device)
        tail_tokens_tensor.requires_grad_(False)
        head_roberta= self.pre_training_model(head_tokens_tensor.to(current_device),
                                              token_type_ids=None,
                                              attention_mask=(head_tokens_tensor.to(current_device) > 0)) #head_roberta.pooler_output.shape  （batch_size,768）
        tail_roberta = self.pre_training_model(tail_tokens_tensor.to(current_device),
                                               token_type_ids=None,
                                               attention_mask=(tail_tokens_tensor.to(current_device) > 0))
        # entity的embedding
        head_embedding= torch.unsqueeze(head_roberta.pooler_output,dim=1)                       #(batch_size,1,768)oxiangxia
        tail_embedding= torch.unsqueeze(tail_roberta.pooler_output, dim=1)                      #(batch_size,1,768)
        # relation的embedding表示
        relation_embedding= torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)      #(batch_size,1,768)
        #输出向量拼接
        triplet_embedding = torch.cat([head_embedding,tail_embedding,relation_embedding], dim=1)
        output = triplet_embedding                                                              # (batch,3,embedding_dim)
        return output
