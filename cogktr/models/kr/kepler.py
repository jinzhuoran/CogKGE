import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from transformers import RobertaModel
from tqdm import tqdm


class KEPLER(nn.Module):
    def __init__(self, entity_dict_len, relation_dict_len, token_length,embedding_dim,device):
        super(KEPLER, self).__init__()
        self.entity_dict_len = entity_dict_len
        self.relation_dict_len = relation_dict_len
        self.token_length=token_length
        self.embedding_dim = embedding_dim
        self.name = "KEPLER"
        self.lookuptable_E=None
        self.lookuptable_R=None
        self.pre_training_model_name = "roberta-base"
        self.tokenizer=None
        self.pre_training_model=None
        self._init_pre_training(device)
        self.square = embedding_dim ** 0.5
        self.relation_embedding = nn.Embedding(num_embeddings=self.relation_dict_len, embedding_dim=self.embedding_dim)
        self.relation_embedding.weight.data = torch.FloatTensor(self.relation_dict_len, self.embedding_dim).uniform_(
            -6 / self.square, 6 / self.square)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight.data, p=2, dim=1)

    def _init_pre_training(self,device):
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pre_training_model_name)
        self.pre_training_model=RobertaModel.from_pretrained(self.pre_training_model_name).to(device)

    def load_lookuotable(self,lookuptable_E,lookuptable_R,device):
        self.lookuptable_E = lookuptable_E
        self.lookuptable_R = lookuptable_R
        self.device=device
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
            # entity_descriptions_dict[self.lookuptable_E["name"][i]]=encoded_text.to(self.device)
            entity_descriptions_dict[self.lookuptable_E["name"][i]]=encoded_text
            # tokens_list.append(encoded_text.to(self.device))
            tokens_list.append(encoded_text)
        self.lookuptable_E("tokens",entity_descriptions_dict)
        self.tokens = nn.Embedding(num_embeddings=len(self.lookuptable_E), embedding_dim=self.token_length)
        # print( torch.cat(tokens_list, dim=0).to(self.device))
        # print( torch.cat(tokens_list, dim=0).shape)
        # self.tokens.weight.data = torch.cat(tokens_list, dim=0).to(self.device)
        self.tokens.weight.data = torch.cat(tokens_list, dim=0).to(self.device)
        self.lookuptable_E.print_table(2)
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
        with torch.no_grad():
            head_tokens_tensor=self.tokens(triplet_idx[:, 0])
            tail_tokens_tensor= self.tokens(triplet_idx[:, 2])
        head_roberta= self.pre_training_model(head_tokens_tensor,
                                              token_type_ids=None,
                                              attention_mask=(head_tokens_tensor > 0)) #这块多卡有问题  #head_roberta.pooler_output.shape  （batch_size,768）
        tail_roberta = self.pre_training_model(tail_tokens_tensor,
                                               token_type_ids=None,
                                               attention_mask=(tail_tokens_tensor > 0))
        # head_roberta= self.pre_training_model(head_tokens_tensor.to(self.device),
        #                                       token_type_ids=None,
        #                                       attention_mask=(head_tokens_tensor .to(self.device)> 0)) #这块多卡有问题  #head_roberta.pooler_output.shape  （batch_size,768）
        # tail_roberta = self.pre_training_model(tail_tokens_tensor.to(self.device),
        #                                        token_type_ids=None,
        #                                        attention_mask=(tail_tokens_tensor .to(self.device)> 0)) #head_roberta.pooler_output.shape  （batch_size,768）
        #idx转token
        # head_tokens_tensor=torch.ones(len(triplet_idx[:,0]),self.token_length).type(torch.cuda.LongTensor).to(self.device)   #（batch_size,token_len）
        # tail_tokens_tensor=torch.ones(len(triplet_idx[:,0]),self.token_length).type(torch.cuda.LongTensor).to(self.device)    #（batch_size,token_len）
        import copy
        # for i in range(len(triplet_idx[:,0])):
            # x=triplet_idx[i, 0].clone().detach()
            # y=triplet_idx[i, 2].clone().detach()
            # head_tokens_tensor[i]=self.lookuptable_E["tokens"][x].type(torch.cuda.LongTensor).to(self.device)
            # tail_tokens_tensor[i]=self.lookuptable_E["tokens"][y].type(torch.cuda.LongTensor).to(self.device)

        # print(head_tokens_tensor)
        # print(head_tokens_tensor.shape)

            # def encode_text(texts):
        #     encoded_texts = list()
        #     for text in texts:
        #         print(text)
        #         encoded_text = tokenizer.encode(
        #             text,
        #             add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
        #             max_length=20,  # 设定最大文本长度
        #             padding="max_length",
        #             return_tensors='pt'  # 返回的类型为pytorch tensor
        #         )
        #         encoded_texts.append(encoded_text)
        #     encoded_texts_tensor = torch.cat(encoded_texts, dim=0)
        #     return encoded_texts_tensor
        #
        # head_tokens_tensor = encode_text(texts)
        # head_tokens_tensor = encode_text(texts)
        # # robert对entity的文本表示
        # print(head_tokens_tensor.to(self.device))
        # head_indexes=triplet_idx[:,0]
        # tail_indexes = triplet_idx[:, 2]
        # head_tokens_list=self.lookuptable_E.batch_index2categort(batch_data=head_indexes, category="tokens")
        # tail_tokens_list = self.lookuptable_E.batch_index2categort(batch_data=tail_indexes, category="tokens")
        # head_tokens_tensor=torch.cat(head_tokens_list, dim=0).to(self.device)    #（batch_size,token_len）
        # tail_tokens_tensor = torch.cat(tail_tokens_list, dim=0).to(self.device)  #（batch_size,token_len）

        # # robert对entity的文本表示
        # head_roberta= self.pre_training_model(head_tokens_tensor.to(self.device),   #head_roberta.pooler_output.shape  （batch_size,768）
        #                                       token_type_ids=None,                  #这块多卡有问题
        #                                       attention_mask=head_tokens_tensor.to(self.device) > 0)
        # tail_roberta = self.pre_training_model(head_tokens_tensor.to(self.device),  #head_roberta.pooler_output.shape  （batch_size,768）
        #                                        token_type_ids=None,
        #                                        attention_mask=tail_tokens_tensor.to(self.device) > 0)
        # entity的embedding
        head_embedding= torch.unsqueeze(head_roberta.pooler_output,dim=1)  #(batch_size,1,768)oxiangxia
        # print("head_embedding",head_embedding)
        tail_embedding= torch.unsqueeze(tail_roberta.pooler_output, dim=1)  #(batch_size,1,768)
        # print("tail_embedding", tail_embedding)
        # relation的embedding表示
        relation_embedding= torch.unsqueeze(self.relation_embedding(triplet_idx[:, 1]), 1)  #(batch_size,1,768)
        # print("relation_embedding", relation_embedding)
        #输出向量拼接
        triplet_embedding = torch.cat([head_embedding,tail_embedding,relation_embedding], dim=1)
        output = triplet_embedding    # (batch,3,embedding_dim)
        return output
