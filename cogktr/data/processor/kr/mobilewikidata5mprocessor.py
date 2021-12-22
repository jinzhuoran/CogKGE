from .baseprocessor import BaseProcessor
from transformers import RobertaTokenizer
from transformers import RobertaModel
from tqdm import tqdm
import torch


class MOBILEWIKIDATA5MProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut):
        """
        :param luts:node_lut,relation_lut
        """
        super().__init__(node_lut, relation_lut)
        self.node_lut = node_lut
        self.pre_training_model_name = "roberta-base"
        self.token_length = 10
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pre_training_model_name)
        self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)

    def process(self, data):
        tokens_list = []
        masks_list = []
        for i in tqdm(range(len(self.node_lut))):
            encoded_text = self.tokenizer.encode_plus(
                self.node_lut["descriptions"][i],
                add_special_tokens=True,
                max_length=self.token_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            tokens_list.append(encoded_text["input_ids"])
            masks_list.append(encoded_text['attention_mask'])
        self.node_lut.add_column(tokens_list, "input_ids")
        self.node_lut.add_column(masks_list, "attention_mask")

        head_input_ids = []
        head_attention_mask = []
        tail_intput_ids = []
        tail_attention_mask = []
        for i in tqdm(range(len(data))):
            head, tail = data[i]["head"], data[i]["tail"]
            head_input_ids.append(self.node_lut.search(head, "input_ids"))
            head_attention_mask.append(self.node_lut.search(head, "attention_mask"))
            tail_intput_ids.append(self.node_lut.search(tail, "input_ids"))
            tail_attention_mask.append(self.node_lut.search(tail, "attention_mask"))
        descriptions = [torch.cat(l, dim=0) for l in
                        [head_input_ids, tail_intput_ids, head_attention_mask, tail_attention_mask]]

        data = self._datable2numpy(data)

        return Cog_Dataset(data, task='kr', descriptions=descriptions)


# from ...dataset import Cog_Dataset
# from tqdm import tqdm
# from transformers import logging
# logging.set_verbosity_error()
from transformers import RobertaTokenizer
# import numpy as np
#
# class MOBILEWIKIDATA5MProcessor:
#     def __init__(self,lut_E,lut_R):
#         self.lut_E=lut_E
#         self.lut_R=lut_R
#     def process(self,datable):
#         datable=self.relation_str2number(datable)
#         datable.print_table(5)
#         datable=self.entity_str2descriptions(datable)
#         datable.print_table(5)
#         datable=self.entity_description_tokenization(datable)
#         datable.print_table(5)
#         dataset=Cog_Dataset(data=datable,task="kr",add_texts=True)
#         return dataset
#     def relation_str2number(self,datable):
#         for i in range(len(datable)):
#             datable["relation"][i]=np.ones((300,))*self.lut_R.str_dic[datable["relation"][i]]
#         return datable
#     def entity_str2descriptions(self,datable):
#         for i in range(len(datable)):
#             datable["head"][i]=self.lut_E["descriptions"][self.lut_E.str_dic[datable["head"][i]]]
#             datable["tail"][i] =self.lut_E["descriptions"][self.lut_E.str_dic[datable["tail"][i]]]
#         return datable
#     def entity_description_tokenization(self,datable):
#         model_name="roberta-base"
#         tokenizer=RobertaTokenizer.from_pretrained(model_name)
#         print("Descriptions Tokenization ... ")
#         for i in tqdm(range(len(datable))):
#             encoded_text_head=tokenizer.encode(
#                 datable["head"][i],
#                 add_special_tokens=True,
#                 max_length=300,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt"
#             )
#             encoded_text_tail = tokenizer.encode(
#                 datable["tail"][i],
#                 add_special_tokens=True,
#                 max_length=300,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt"
#             )
#             datable["head"][i]=encoded_text_head[0].data.numpy()
#             datable["tail"][i]=encoded_text_tail[0].data.numpy()
#         return datable
from ...dataset import Cog_Dataset
# class MOBILEWIKIDATA5MProcessor:
#     def __init__(self,lut_E,lut_R):
#         self.lut_E=lut_E
#         self.lut_R=lut_R
#     def process(self,datable):
#         datable=self.str2number(datable)
#         dataset=Cog_Dataset(data=datable,task="kr")
#         return dataset
#     def str2number(self,datable):
#         for i in range(len(datable)):
#             datable["head"][i]=self.lut_E.str_dic[datable["head"][i]]
#             datable["relation"][i]=self.lut_R.str_dic[datable["relation"][i]]
#             datable["tail"][i]=self.lut_E.str_dic[datable["tail"][i]]
#         return datable
