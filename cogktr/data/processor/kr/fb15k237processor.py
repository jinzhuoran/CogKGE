# from ...loader import *
# import numpy as np
# from ...lut import LUT
# from ...dataset import Feeder
#
# class FB15K237Processor:
#     def __init__(self,lookUpTable:LUT):
#         self.lut = lookUpTable
#
#     def process(self,data):
#         """
#         convert list of string triples to corresponding datasets
#         return: constructed dataset containing __len__ and __getitem__ methods
#         """
#         return Feeder(self.list2numpy(data))
#
#
#     def list2numpy(self,data):
#         """
#         data: triples in string form(entity names and relation names)
#         return: triples in numpy form
#         """
#         heads=list()
#         relations=list()
#         tails=list()
#
#         for i in range(len(data[0])):
#             heads.append(self.lut.entity2id(data[0][i]))
#             relations.append(self.lut.relation2id(data[1][i]))
#             tails.append(self.lut.entity2id(data[2][i]))
#
#         heads_np=np.array(heads,dtype=np.int64)[:,np.newaxis]
#         relations_np=np.array(relations,dtype=np.int64)[:,np.newaxis]
#         tails_np=np.array(tails,dtype=np.int64)[:,np.newaxis]
#         data_numpy=np.hstack((heads_np,relations_np,tails_np))
#
#         return data_numpy
#
from ...dataset import Cog_Dataset
class FB15K237Processor:
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
