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
class MOBILEWIKIDATA5MProcessor:
    def __init__(self,lut_E,lut_R):
        self.lut_E=lut_E
        self.lut_R=lut_R
    def process(self,datable):
        datable=self.str2number(datable)
        dataset=Cog_Dataset(data=datable,task="kr")
        return dataset
    def str2number(self,datable):
        for i in range(len(datable)):
            datable["head"][i]=self.lut_E.str_dic[datable["head"][i]]
            datable["relation"][i]=self.lut_R.str_dic[datable["relation"][i]]
            datable["tail"][i]=self.lut_E.str_dic[datable["tail"][i]]
        return datable


