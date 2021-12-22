import os
import pickle

import numpy as np
import pandas as pd

from ...datable import Datable
from ...lut import LookUpTable
from ....utils.download_utils import Download_Data
from ...vocabulary import Vocabulary
import json

from .baseloader import BaseLoader


class EVENTKG2MLoader(BaseLoader):
    def __init__(self, path, download=False, download_path=None):
        super().__init__(path, download, download_path,
                         train_name="eventkg2m_train.txt",
                         valid_name="eventkg2m_valid.txt",
                         test_name="eventkg2m_test.txt")
        self.time_vocab = Vocabulary()
        self.entity_lut_name = "eventkg2m_entities_lut.json"
        self.event_lut_name = "eventkg2m_events_lut.json"
        self.relation_lut_name = "eventkg2m_relations_lut.json"

    def _load_data(self, path):
        return BaseLoader._load_data(self, path=path, column_names=["head", "relation", "tail", "start", "end"])

    def download_action(self):
        self.downloader.EVENTKG2M()

    def _build_vocabs(self, train_data, valid_data, test_data):
        BaseLoader._build_vocabs(self, train_data, valid_data, test_data)
        self.time_vocab.buildVocab(train_data['start'].tolist(), train_data['end'].tolist(),
                                   valid_data['start'].tolist(), valid_data['end'].tolist(),
                                   test_data['start'].tolist(), test_data['end'].tolist())

    def load_all_vocabs(self, ):
        return self.node_vocab, self.relation_vocab, self.time_vocab

    def save_vocabs_to_pickle(self, file_name):
        with open(file_name, "wb") as file:
            pickle.dump([self.node_vocab, self.relation_vocab, self.time_vocab], file, pickle.HIGHEST_PROTOCOL)

    def load_vocabs_from_pickle(self, file_name):
        with open(file_name, "rb") as file:
            self.node_vocab, self.relation_vocab, self.time_vocab = pickle.load(file)

    def _load_lut(self, path):
        total_path = os.path.join(self.path, path).replace('\\', '/')
        lut = LookUpTable()
        lut.read_json(total_path)
        lut.transpose()
        return lut

    def load_relation_lut(self):
        preprocessed_file = os.path.join(self.path, "relation_lut.pkl")
        if os.path.exists(preprocessed_file):
            relation_lut = LookUpTable()
            relation_lut.read_from_pickle(preprocessed_file)
        else:
            relation_lut = self._load_lut(self.relation_lut_name)
            relation_lut.add_vocab(self.relation_vocab)
            relation_lut.save_to_pickle(preprocessed_file)
        return relation_lut

    def load_node_lut(self):
        preprocessed_file = os.path.join(self.path, "node_lut.pkl")
        if os.path.exists(preprocessed_file):
            node_lut = LookUpTable()
            node_lut.read_from_pickle(preprocessed_file)
            # node_lut = pd.read_pickle(preprocessed_file)
        else:
            entity_lut = self._load_lut(self.entity_lut_name)
            entity_lut.rename(columns={"entity_label": "node_label", "entity_rdf": "node_rdf"})
            entity_lut.add_column(['entity'] * len(entity_lut), "node_type")
            # entity_lut = entity_lut.rename(columns={"entity_label": "node_label", "entity_rdf": "node_rdf"})
            # entity_lut = entity_lut.assign(node_type=pd.Series(['entity'] * len(entity_lut)).values)

            event_lut = self._load_lut(self.event_lut_name)
            event_lut.rename(columns={"event_label": "node_label", "event_rdf": "node_rdf"})
            event_lut.add_column(['event'] * len(event_lut), "node_type")

            # event_lut = event_lut.rename(columns={"event_label": "node_label", "event_rdf": "node_rdf"})
            # event_lut = event_lut.assign(node_type=pd.Series(['event'] * len(event_lut)).values)

            node_lut = entity_lut.append(event_lut)
            node_lut.add_vocab(self.node_vocab)
            node_lut.save_to_pickle(preprocessed_file)
            # node_lut.to_pickle(preprocessed_file)
        return node_lut

    def load_time_lut(self):
        time_lut = LookUpTable()
        time_lut.add_vocab(self.time_vocab)
        return time_lut

    def load_all_lut(self):
        node_lut = self.load_node_lut()
        relation_lut = self.load_relation_lut()
        time_lut = self.load_time_lut()
        return node_lut, relation_lut,time_lut

# class EVENTKG2MLoader:
#     """
#     DataLoader for EventKG2M dataset
#     1. load_all_data return the train,valid,test dataset in order and construct the Vocabulary at the same time
#        so make sure to load the data before using the Vocab
#     2. Only three APIs are exposed:
#     (1) load_all_data return the datable object
#     (2) load_all_vocab return the vocab constructed from the raw data
#     (3) load_all_LUT return the lookup table generated from the json file
#     """
#
#     def __init__(self, path, download=False, download_path=None):
#         self.path = path
#         self.download = download
#         self.download_path = download_path
#         self.node_vocab = Vocabulary()
#         self.relation_vocab = Vocabulary()
#         self.time_vocab = Vocabulary()
#
#         if self.download:
#             downloader = Download_Data(dataset_path=self.download_path)
#             downloader.EVENTKG2M()
#         self.train_name = "eventkg2m_train.txt"
#         self.valid_name = "eventkg2m_valid.txt"
#         self.test_name = "eventkg2m_test.txt"
#         self.entity_lut_name = "eventkg2m_entities_lut.json"
#         self.event_lut_name = "eventkg2m_events_lut.json"
#         # self.entity_lut_name = "entity_short.json"
#         # self.event_lut_name = "event_short.json"
#         self.relation_lut_name = "eventkg2m_relations_lut.json"
#
#     def _load_data(self, path):
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         datable = Datable()
#         datable.read_csv(total_path, sep='\t', names=['head', 'relation', 'tail', 'start', 'end'])
#         # df = pd.read_csv(total_path, sep='\t', names=['head', 'relation', 'tail', 'start', 'end'])
#         return datable
#
#     def _load_lut(self, path):
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         df = pd.read_json(total_path)
#         return df.T
#
#     def load_node_lut(self):
#         preprocessed_file = os.path.join(self.path, "node_lut.pkl")
#         if os.path.exists(preprocessed_file):
#             node_lut = pd.read_pickle(preprocessed_file)
#         else:
#             entity_lut = self._load_lut(self.entity_lut_name)
#             entity_lut = entity_lut.rename(columns={"entity_label": "node_label", "entity_rdf": "node_rdf"})
#             entity_lut = entity_lut.assign(node_type=pd.Series(['entity'] * len(entity_lut)).values)
#
#             event_lut = self._load_lut(self.event_lut_name)
#             event_lut = event_lut.rename(columns={"event_label": "node_label", "event_rdf": "node_rdf"})
#             event_lut = event_lut.assign(node_type=pd.Series(['event'] * len(event_lut)).values)
#
#             node_lut = entity_lut.append(event_lut)
#             node_lut.to_pickle(preprocessed_file)
#         return node_lut
#
#     def load_relation_lut(self):
#         preprocessed_file = os.path.join(self.path, "relation_lut.pkl")
#         if os.path.exists(preprocessed_file):
#             relation_lut = pd.read_pickle(preprocessed_file)
#         else:
#             relation_lut = self._load_lut(self.relation_lut_name)
#         return relation_lut
#
#     def load_all_lut(self):
#         node_lut = self.load_node_lut()
#         relation_lut = self.load_relation_lut()
#         return node_lut, relation_lut
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
#         pre_train = os.path.join(self.path, "train_data.pkl")
#         pre_valid = os.path.join(self.path, "valid_data.pkl")
#         pre_test = os.path.join(self.path, "test_data.pkl")
#         pre_vocab = os.path.join(self.path, "vocab.pkl")
#
#         if os.path.exists(pre_train) and os.path.exists(pre_valid) and os.path.exists(pre_test) and os.path.exists(pre_vocab):
#             train_data,valid_data,test_data = Datable(),Datable(),Datable()
#             train_data.read_from_pickle(pre_train)
#             valid_data.read_from_pickle(valid_data)
#             test_data.read_from_pickle(test_data)
#             with open(pre_vocab,"rb") as file:
#                 self.node_vocab, self.relation_vocab, self.time_vocab = pickle.load(file)
#
#         else:
#             train_data = self._load_data(self.train_name)
#             valid_data = self._load_data(self.valid_name)
#             test_data = self._load_data(self.test_name)
#             train_data.save_to_pickle(pre_train)
#             test_data.save_to_pickle(pre_test)
#
#             self._build_vocabs(train_data, valid_data, test_data)
#             with open(pre_vocab, "wb") as file:
#                 pickle.dump([self.node_vocab, self.relation_vocab, self.time_vocab], file, pickle.HIGHEST_PROTOCOL)
#
#         return train_data, valid_data, test_data
#

#
#     def load_all_vocabs(self, ):
#         """
#         load the built vocabulary
#         !!  Must be used after calling the load_all_data function
#             Since the vocabs are built on the data read from the files. !!
#         :return: node_vocab,relation_vocab,time_vocab
#         """
#         return self.node_vocab, self.relation_vocab, self.time_vocab
#
#     def _build_vocabs(self, train_data, valid_data, test_data):
#         """
#         build vocaulary according to the data
#         :param train_data: dataframe
#         :param valid_data: dataframe
#         :param test_data: dataframe
#         :return: None
#         """
#         self.node_vocab.buildVocab(train_data['head'].tolist(), train_data['tail'].tolist(),
#                                    valid_data['head'].tolist(), valid_data['tail'].tolist(),
#                                    test_data['head'].tolist(), test_data['tail'].tolist())
#         self.relation_vocab.buildVocab(train_data['relation'].tolist(),
#                                        valid_data['relation'].tolist(),
#                                        test_data['relation'].tolist())
#         self.time_vocab.buildVocab(train_data['start'].tolist(), train_data['end'].tolist(),
#                                    valid_data['start'].tolist(), valid_data['end'].tolist(),
#                                    test_data['start'].tolist(), test_data['end'].tolist())
#
#     def _dataset2numpy(self, data):
#         """
#         convert a dataframe to numpy array form according to the previously constructed Vocab
#         :param data: dataframe (dataset_len,5)
#         :return: numpy array
#         """
#         data['head'] = self._series2numpy(data['head'], self.node_vocab)
#         data['tail'] = self._series2numpy(data['tail'], self.node_vocab)
#         data['relation'] = self._series2numpy(data['relation'], self.relation_vocab)
#         data['start'] = self._series2numpy(data['start'], self.time_vocab)
#         data['end'] = self._series2numpy(data['end'], self.time_vocab)
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
#         valid_data.save_to_pickle(pre_valid)


# class EVENTKGLoader:
#     def __init__(self, path,download=False,download_path=None):
#         self.path = path
#         self.download = download
#         self.download_path=download_path
#         self.entity_list = list()
#         self.relation_list = list()
#         if self.download == True:
#             downloader = Download_Data(dataset_path=self.download_path)
#             downloader.EVENTKG()
#         self.train_name="data_all.txt"
#         self.valid_name="temp_1.txt"
#         self.test_name="temp_2.txt"


#     def _load_data(self, path):
#         heads = []
#         relations = []
#         tails = []
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         with open(total_path) as file:
#             for line in file:
#                 h, r, t = line.strip().split("\t")
#                 heads.append(h)
#                 relations.append(r)
#                 tails.append(t)
#                 self.entity_list.append(h)
#                 self.entity_list.append(t)
#                 self.relation_list.append(r)
#         datable = Datable()
#         datable(["head", "relation", "tail"], [heads, relations, tails])
#         return datable

#     def load_train_data(self):
#         train_data = self._load_data(self.train_name)
#         return train_data

#     def load_valid_data(self):
#         valid_data = self._load_data(self.valid_name)
#         return valid_data

#     def load_test_data(self):
#         test_data = self._load_data(self.test_name)
#         return test_data

#     def load_all_data(self):
#         train_data = self._load_data(self.train_name)
#         valid_data = self._load_data(self.valid_name)
#         test_data = self._load_data(self.test_name)
#         return train_data, valid_data, test_data

#     def _load_lut(self, path, category=None):
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         if not os.path.exists(total_path):
#             if category == "entity":
#                 print("Creating entities.json...")
#                 entity_name_list = list(set(list(self.entity_list)))
#                 # entity_name_list.sort(key=list(self.entity_list).index)
#                 lookuptable = LookUpTable()
#                 lookuptable.create_table(create_dic=True, item_list=entity_name_list)
#                 entities_dict = dict()
#                 for i in range(len(lookuptable)):
#                     entities_dict[lookuptable["name"][i]] = i
#                 json.dump(entities_dict, open(total_path, "w"), indent=4, sort_keys=True)

#             if category == "relation":
#                 print("Creating relations.json...")
#                 relation_name_list = list(set(list(self.relation_list)))
#                 # relation_name_list.sort(key=list(self.relation_list).index)
#                 lookuptable = LookUpTable()
#                 lookuptable.create_table(create_dic=True, item_list=relation_name_list)
#                 relations_dict = dict()
#                 for i in range(len(lookuptable)):
#                     relations_dict[lookuptable["name"][i]] = i
#                 json.dump(relations_dict, open(total_path, "w"), indent=4, sort_keys=True)

#         if category == "entity":
#             with open(total_path) as file:
#                 entity2idx = json.load(file)
#             lookuptable = LookUpTable()
#             lookuptable.create_table(create_dic=False, str_dic=entity2idx)
#             return lookuptable
#         if category == "relation":
#             with open(total_path) as file:
#                 relation2idx = json.load(file)
#             lookuptable = LookUpTable()
#             lookuptable.create_table(create_dic=False, str_dic=relation2idx)
#             return lookuptable

#     def load_entity_lut(self):
#         entity2idx = self._load_lut(path="entities.json", category="entity")
#         return entity2idx

#     def load_relation_lut(self):
#         relation2idx = self._load_lut(path="relations.json", category="realtion")
#         return relation2idx

#     def load_all_lut(self):
#         entity2idx = self._load_lut(path="entities.json", category="entity")
#         relation2idx = self._load_lut(path="relations.json", category="relation")
#         return entity2idx, relation2idx
