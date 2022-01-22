import json
from ...utils.download_utils import Download_Data
from ..vocabulary import Vocabulary
import os
import pandas as pd
import numpy as np
import pickle
from ..datable import Datable
from ..lut import LookUpTable
import prettytable as pt


class BaseLoader:
    def __init__(self, dataset_path, download,raw_data_path,processed_data_path,train_name, valid_name, test_name,data_name):
        self.dataset_path = dataset_path
        self.download = download
        self.raw_data_path=os.path.join(dataset_path,raw_data_path)
        self.processed_data_path=os.path.join (dataset_path,processed_data_path)
        self.train_name = train_name
        self.valid_name = valid_name
        self.test_name = test_name
        self.data_name=data_name
        if self.download:
            self.downloader = Download_Data(dataset_path=self.dataset_path)
            self.download_action()
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)

        self.node_vocab = Vocabulary()
        self.relation_vocab = Vocabulary()

    def download_action(self):
        raise NotImplementedError("Download Action Must Be Specified!")

    def _load_data(self, path, data_type=None ,column_names=None):
        if column_names is None:
            column_names = ["head", "relation", "tail"]
        total_path = os.path.join(self.raw_data_path, path)
        datable = Datable(data_type)
        datable.read_csv(total_path, sep='\t', names=column_names)
        return datable

    def load_train_data(self):
        train_data = self._load_data(self.train_name,data_type="train")
        self.train_len=len(train_data)
        return train_data

    def load_valid_data(self):
        valid_data = self._load_data(self.valid_name,data_type="valid")
        self.valid_len=len(valid_data)
        return valid_data

    def load_test_data(self):
        test_data = self._load_data(self.test_name,data_type="test")
        self.test_len=len(test_data)
        return test_data

    def load_all_data(self):
        pre_train = os.path.join(self.processed_data_path, "train_data.pkl")
        pre_valid = os.path.join(self.processed_data_path, "valid_data.pkl")
        pre_test = os.path.join(self.processed_data_path,"test_data.pkl")
        pre_vocab = os.path.join(self.processed_data_path,"vocab.pkl")

        if os.path.exists(pre_train) and os.path.exists(pre_valid) \
                and os.path.exists(pre_test) and os.path.exists(pre_vocab):
            train_data,valid_data,test_data = Datable(data_type="train"),Datable(data_type="valid"),Datable(data_type="test")
            train_data.read_from_pickle(pre_train)
            valid_data.read_from_pickle(pre_valid)
            test_data.read_from_pickle(pre_test)
            self.train_len = len(train_data)
            self.valid_len = len(valid_data)
            self.test_len = len(test_data)
            self.load_vocabs_from_pickle(pre_vocab)

        else:
            train_data = self.load_train_data()
            valid_data = self.load_valid_data()
            test_data = self.load_test_data()
            train_data.save_to_pickle(pre_train)
            valid_data.save_to_pickle(pre_valid)
            test_data.save_to_pickle(pre_test)

            self._build_vocabs(train_data, valid_data, test_data)
            self.save_vocabs_to_pickle(pre_vocab)
            # with open(pre_vocab, "wb") as file:
            #     pickle.dump([self.node_vocab,self.relation_vocab],file,pickle.HIGHEST_PROTOCOL)

        return train_data, valid_data, test_data

    def save_vocabs_to_pickle(self,file_name):
        with open(file_name, "wb") as file:
            pickle.dump([self.node_vocab,self.relation_vocab],file,pickle.HIGHEST_PROTOCOL)

    def load_vocabs_from_pickle(self,file_name):
        with open(file_name,"rb") as file:
            self.node_vocab,self.relation_vocab = pickle.load(file)

    def load_all_lut(self):
        node_lut = LookUpTable()
        node_lut.add_vocab(self.node_vocab)
        node_lut.add_processed_path(self.processed_data_path)
        relation_lut = LookUpTable()
        relation_lut.add_vocab(self.relation_vocab)
        relation_lut.add_processed_path(self.processed_data_path)
        return node_lut,relation_lut

    def load_all_vocabs(self, ):
        """
        load the built vocabulary
        !!  Must be used after calling the load_all_data function
            Since the vocabs are built on the data read from the files. !!
        :return: node_vocab,relation_vocab,time_vocab
        """
        return self.node_vocab, self.relation_vocab

    def _build_vocabs(self, train_data, valid_data, test_data):
        """
        build vocaulary according to the data
        :param train_data: datable
        :param valid_data: datable
        :param test_data: datable
        :return: None
        """
        self.node_vocab.buildVocab(train_data['head'].tolist(), train_data['tail'].tolist(),
                                   valid_data['head'].tolist(), valid_data['tail'].tolist(),
                                   test_data['head'].tolist(), test_data['tail'].tolist())
        self.relation_vocab.buildVocab(train_data['relation'].tolist(),
                                       valid_data['relation'].tolist(),
                                       test_data['relation'].tolist())

    def describe(self):
        tb = pt.PrettyTable()
        tb.field_names = [self.data_name,"train","valid","test","node","relation"]
        tb.add_row(["num",self.train_len,self.valid_len,self.test_len,len(self.node_vocab),len(self.relation_vocab)])
        print(tb)
# class BaseLoader:
#     def __init__(self, path, download, download_path, train_name, valid_name, test_name):
#         self.path = path
#         self.download = download
#         self.download_path = download_path
#         self.entity_vocab = Vocabulary()
#         self.relation_vocab = Vocabulary()
#         if self.download:
#             self.downloader = Download_Data(dataset_path=self.download_path)
#             self.download_action()
#         self.train_name = train_name
#         self.valid_name = valid_name
#         self.test_name = test_name
#
#     def download_action(self):
#         raise NotImplementedError("Download Action Must Be Specified!")
#
#     def _load_data(self, path):
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         df = pd.read_csv(total_path, sep='\t', names=['head', 'relation', 'tail'])
#         return df
#
#     def load_train_data(self):
#         train_data = self._load_data(self.train_name)
#         return train_data
#
#     def load_valid_data(self):
#         valid_data = self._load_data(self.valid_name)
#         return valid_data
#
#     def load_test_data(self):
#         test_data = self._load_data(self.test_name)
#         return test_data
#
#     def load_all_data(self):
#         pre_train = os.path.join(self.path, "train_data.npy")
#         pre_valid = os.path.join(self.path, "valid_data.npy")
#         pre_test = os.path.join(self.path, "test_data.npy")
#         pre_vocab = os.path.join(self.path, "vocab.pkl")
#
#         if os.path.exists(pre_train) and os.path.exists(pre_valid) and os.path.exists(pre_test) and os.path.exists(
#                 pre_vocab):
#             train_data = np.load(pre_train)
#             valid_data = np.load(pre_valid)
#             test_data = np.load(pre_test)
#             with open(pre_vocab, "rb") as file:
#                 self.entity_vocab, self.relation_vocab = pickle.load(file)
#         else:
#             train_data = self._load_data(self.train_name)
#             valid_data = self._load_data(self.valid_name)
#             test_data = self._load_data(self.test_name)
#             self._build_vocabs(train_data, valid_data, test_data)
#             train_data, valid_data, test_data = [self._dataset2numpy(any_data)
#                                                  for any_data in [train_data, valid_data, test_data]]
#
#             np.save(pre_train, train_data)
#             np.save(pre_valid, valid_data)
#             np.save(pre_test, test_data)
#             with open(pre_vocab, "wb") as file:
#                 pickle.dump([self.entity_vocab, self.relation_vocab], file, pickle.HIGHEST_PROTOCOL)
#
#         return train_data, valid_data, test_data
#
#     def load_all_vocabs(self, ):
#         """
#         load the built vocabulary
#         !!  Must be used after calling the load_all_data function
#             Since the vocabs are built on the data read from the files. !!
#         :return: entity_vocab,relation_vocab
#         """
#         return self.entity_vocab, self.relation_vocab
#
#     def _build_vocabs(self, train_data, valid_data, test_data):
#         """
#         build vocaulary according to the data
#         :param train_data: dataframe
#         :param valid_data: dataframe
#         :param test_data: dataframe
#         :return: None
#         """
#         self.entity_vocab.buildVocab(train_data['head'].tolist(), train_data['tail'].tolist(),
#                                      valid_data['head'].tolist(), valid_data['tail'].tolist(),
#                                      test_data['head'].tolist(), test_data['tail'].tolist())
#         self.relation_vocab.buildVocab(train_data['relation'].tolist(),
#                                        valid_data['relation'].tolist(),
#                                        test_data['relation'].tolist())
#
#     def _dataset2numpy(self, data):
#         """
#         convert a dataframe to numpy array form according to the previously constructed Vocab
#         :param data: dataframe (dataset_len,5)
#         :return: numpy array
#         """
#         data['head'] = self._series2numpy(data['head'], self.entity_vocab)
#         data['tail'] = self._series2numpy(data['tail'], self.entity_vocab)
#         data['relation'] = self._series2numpy(data['relation'], self.relation_vocab)
#         return data.to_numpy()
#
#     @staticmethod
#     def _series2numpy(series, vocab):
#         """
#         convert a dataframe containing str-type elements to a numpy array using word2idx
#         :param series: pandas data series
#         :param vocab: corresponding word2idx function
#         :return: numpy array
#         """
#         # print("Hello World!")
#         word2idx = vocab.getWord2idx()
#         f = lambda word: word2idx[word]
#         return series.apply(f)
