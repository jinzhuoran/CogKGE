from ...dataset import Cog_Dataset
from .baseprocessor import BaseProcessor


class EVENTKG2MProcessor(BaseProcessor):
    def __init__(self, node_vocab, relation_vocab, time_vocab):
        """
        :param vocabs: node_vocab,relation_vocab,time_vocab
        """
        super().__init__(node_vocab,relation_vocab)
        self.time_vocab = time_vocab

    # def process(self, data):
    #     data = self._datable2numpy(data)
    #     return Cog_Dataset(data, task='kr')

    def _datable2numpy(self, data):
        """
        convert a datable to numpy array form according to the previously constructed Vocab
        :param data: datable (dataset_len,5)
        :return: numpy array
        """
        data.str2idx("head",self.node_vocab)
        data.str2idx("tail",self.node_vocab)
        data.str2idx("relation",self.relation_vocab)
        data.str2idx("start",self.time_vocab)
        data.str2idx("end",self.time_vocab)
        return data.to_numpy()



# from ...dataset import Cog_Dataset
# class EVENTKGProcessor:
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
