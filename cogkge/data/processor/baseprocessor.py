import copy
import os
import pickle

from ..dataset import Cog_Dataset


class BaseProcessor:
    def __init__(self, data_name, node_lut, relation_lut, reprocess=True,
                 time=None, nodetype=None, description=None, graph=None):
        """
        :param vocabs: node_vocab,relation_vocab from node_lut relation_lut
        """
        self.data_name = data_name
        self.node_vocab = node_lut.vocab
        self.relation_vocab = relation_lut.vocab
        self.node_lut = node_lut
        self.relation_lut = relation_lut
        self.reprocess = reprocess
        self.time = time
        self.nodetype = nodetype
        self.description = description
        self.graph = graph
        self.processed_path = node_lut.processed_path
        # self.node_vocab = node_vocab
        # self.relation_vocab = relation_vocab

    def process(self, data):
        path = os.path.join(self.processed_path, "{}_dataset.pkl".format(data.data_type))
        if os.path.exists(path) and not self.reprocess:
            print("load {} dataset".format(data.data_type))
            with open(path, "rb") as new_file:
                new_data = pickle.loads(new_file.read())
            return new_data
        else:
            data = self._datable2numpy(data)
            dataset = Cog_Dataset(data, task='kr')
            dataset.data_name = self.data_name
            file = open(path, "wb")
            file.write(pickle.dumps(dataset))
            file.close()

            return dataset

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

    def process_lut(self):
        return self.node_lut, self.relation_lut

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
