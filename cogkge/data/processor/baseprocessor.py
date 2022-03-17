import copy
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from ..dataset import Cog_Dataset
import numpy as np


class BaseProcessor:
    def __init__(self, data_name, node_lut, relation_lut, reprocess=True,mode="normal",
                 time=None, nodetype=None, description=None, graph=None,train_pattern="score_based"):
        """
        :param vocabs: node_vocab,relation_vocab from node_lut relation_lut
        """
        self.mode = mode
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
        self.train_pattern=train_pattern
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
            if self.train_pattern=="classification_based":
                triplet_label_dict=self.create_triplet_label(data)
                data=self.convert_label_construct(triplet_label_dict)
            dataset = Cog_Dataset(data, task='kr',train_pattern=self.train_pattern,mode=self.mode)
            dataset.data_name = self.data_name
            if self.train_pattern == "scored_based":
                file = open(path, "wb")
                file.write(pickle.dumps(dataset))
                file.close()
            return dataset

    def convert_label_construct(self,triplet_label_dict):
        h_r_list=list()
        t_list=list()
        print("convert_label_construct...")
        for key,value in tqdm(triplet_label_dict.items()):
            h_r_list.append(np.array(key))
            vector_label=np.zeros((len(self.node_lut)))
            for index in value:
                vector_label[index]=1
            t_list.append(vector_label)

        t=np.array(t_list)
        h_r=np.array(h_r_list)
        return (h_r,t)


    def create_triplet_label(self,data):
        triplet_label_dict=defaultdict(list)
        print("create_triplet_label...")
        for i in tqdm(range(len(data))):
            triplet_h_r=tuple(data[i][:2])
            triplet_t=int(data[i][2].item())
            triplet_label_dict[triplet_h_r].append(triplet_t)
        return triplet_label_dict




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
