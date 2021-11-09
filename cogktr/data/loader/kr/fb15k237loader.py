# import os
# from ...lut import LUT
# class FB15K237Loader:
#     def __init__(self,path):
#         """
#         load FB15K-237 dataset according to the given raw data path
#         """
#         self.path=path
#
#     def _load_data(self,path):
#         heads=[]
#         relations=[]
#         tails=[]
#         total_path=os.path.join(self.path,path).replace('\\', '/')
#         with open(total_path) as file:
#             for line in file:
#                 #.strip()去除最后的换行符
#                 #.split("\t")把一行用Tab分割成不同的元素
#                 h,r,t=line.strip().split("\t")
#                 heads.append(h)
#                 relations.append(r)
#                 tails.append(t)
#         return [heads,relations,tails]
#
#     def load_train_data(self):
#         train_data=self._load_data("train.txt")
#         return train_data
#
#     def load_valid_data(self):
#         valid_data=self._load_data("train.txt")
#         return valid_data
#
#     def load_test_data(self):
#         test_data=self._load_data("train.txt")
#         return test_data
#
#     def load_all_data(self):
#         """
#         data are all lists of strings which can be further converted to id arrays using processor.process()
#         """
#         train_data=self._load_data("train.txt")
#         valid_data=self._load_data("valid.txt")
#         test_data=self._load_data("test.txt")
#         return train_data,valid_data,test_data
#
#     def _load_dict(self,path):
#         str2idx={}
#         total_path=os.path.join(self.path,path).replace('\\', '/')
#         with open(total_path) as file:
#             for line in file:
#                 #.strip()去除最后的换行符
#                 #.split("\t")把一行用Tab分割成不同的元素
#                 idx,str=line.strip().split("\t")
#                 #读取出来的都是字符型，所以要强制转换为整型
#                 str2idx[str]=int(idx)
#         return str2idx
#
#     def load_entity_dict(self):
#         entity2idx=self._load_dict("entities.dict")
#         return entity2idx
#
#     def load_relation_dict(self):
#         relation2idx=self._load_dict("relations.dict")
#         return relation2idx
#
#     def load_all_dict(self):
#         entity2idx=self._load_dict("entities.dict")
#         relation2idx=self._load_dict("relations.dict")
#         return entity2idx,relation2idx
#
#     def createLUT(self):
#         return LUT(self.load_entity_dict(),self.load_relation_dict())
#


import os
from ...datable import Datable
from ...lut import LookUpTable
class FB15K237Loader:
    def __init__(self,path):
        self.path=path

    def _load_data(self,path):
        heads=[]
        relations=[]
        tails=[]
        total_path=os.path.join(self.path,path).replace('\\', '/')
        with open(total_path) as file:
            for line in file:
                h,r,t=line.strip().split("\t")
                heads.append(h)
                relations.append(r)
                tails.append(t)

        datable = Datable()
        datable(["head", "relation", "tail"], [heads, relations, tails])
        return datable

    def load_train_data(self):
        train_data=self._load_data("train.txt")
        return train_data

    def load_valid_data(self):
        valid_data=self._load_data("train.txt")
        return valid_data

    def load_test_data(self):
        test_data=self._load_data("train.txt")
        return test_data

    def load_all_data(self):
        train_data=self._load_data("train.txt")
        valid_data=self._load_data("valid.txt")
        test_data=self._load_data("test.txt")
        return train_data,valid_data,test_data

    def _load_lut(self,path):
        str2idx={}
        total_path=os.path.join(self.path,path).replace('\\', '/')
        with open(total_path) as file:
            for line in file:
                #.strip()去除最后的换行符
                #.split("\t")把一行用Tab分割成不同的元素
                idx,str=line.strip().split("\t")
                #读取出来的都是字符型，所以要强制转换为整型
                str2idx[str]=int(idx)
        lookuptable=LookUpTable()
        lookuptable.create_table(create_dic=False,str_dic=str2idx)
        return lookuptable

    def load_entity_lut(self):
        entity2idx=self._load_lut("entities.dict")
        return entity2idx

    def load_relation_lut(self):
        relation2idx=self._load_lut("relations.dict")
        return relation2idx

    def load_all_lut(self):
        entity2idx=self._load_lut("entities.dict")
        relation2idx=self._load_lut("relations.dict")
        return entity2idx,relation2idx

