from .baseprocessor import BaseProcessor


class WN18RRProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut):
        super().__init__(node_lut, relation_lut)


# from ...dataset import Cog_Dataset
# class WN18RRProcessor:
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