import copy

from ...dataset import Cog_Dataset


class BaseProcessor:
    def __init__(self, node_lut, relation_lut):
        """
        :param vocabs: node_vocab,relation_vocab from node_lut relation_lut
        """
        self.node_vocab = node_lut.vocab
        self.relation_vocab = relation_lut.vocab
        # self.node_vocab = node_vocab
        # self.relation_vocab = relation_vocab

    def process(self, data):
        data = self._datable2numpy(data)
        return Cog_Dataset(data, task='kr')

    def _datable2numpy(self, data):
        """
        convert a datable to numpy array form according to the previously constructed Vocab
        :param data: datable (dataset_len,5)
        :return: numpy array
        """
        data = copy.deepcopy(data)
        data.str2idx("head", self.node_vocab)
        data.str2idx("tail", self.node_vocab)
        data.str2idx("relation", self.relation_vocab)
        return data.to_numpy()

    @staticmethod
    def _series2numpy(series, vocab):
        """
        convert a dataframe containing str-type elements to a numpy array using word2idx
        :param series: pandas data series
        :param vocab: corresponding word2idx function
        :return: numpy array
        """
        # print("Hello World!")
        word2idx = vocab.getWord2idx()
        f = lambda word: word2idx[word]
        return series.apply(f)
