import os
import pickle

import pandas as pd
import prettytable as pt
from tqdm import tqdm

from .baseloader import BaseLoader
from ..lut import LookUpTable
from ..vocabulary import Vocabulary


class EVENTKG240KLoader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="EVENTKG240K/raw_data",
                         processed_data_path="EVENTKG240K/processed_data",
                         train_name="eventkg240k_train.txt",
                         valid_name="eventkg240k_valid.txt",
                         test_name="eventkg240k_test.txt",
                         data_name="EVENTKG240K")

        self.time_vocab = Vocabulary()
        self.entity_lut_name = "eventkg240k_entities_lut.json"
        self.event_lut_name = "eventkg240k_events_lut.json"
        self.relation_lut_name = "eventkg240k_relations_lut.json"

    def _load_data(self, path, data_type):
        return BaseLoader._load_data(self, path=path, data_type=data_type,
                                     column_names=["head", "relation", "tail", "start", "end"])

    def download_action(self):
        self.downloader.EVENTKG240K()

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
        total_path = os.path.join(self.raw_data_path, path)
        lut = LookUpTable()
        lut.read_json(total_path)
        lut.transpose()
        return lut

    def load_node_lut(self):
        preprocessed_file = os.path.join(self.processed_data_path, "node_lut.pkl")
        if os.path.exists(preprocessed_file):
            node_lut = LookUpTable()
            node_lut.read_from_pickle(preprocessed_file)
        else:
            entity_lut = self._load_lut(self.entity_lut_name)
            entity_lut.add_column(['entity'] * len(entity_lut.data), "node_type")

            event_lut = self._load_lut(self.event_lut_name)
            event_lut.add_column(['event'] * len(event_lut.data), "node_type")


            node_lut = entity_lut.append(event_lut)
            node_lut.add_vocab(self.node_vocab)

            df = pd.DataFrame([self.node_vocab.word2idx]).T
            df = df.rename({0: "name_id"}, axis=1)
            node_lut.data = pd.merge(df, node_lut.data, left_index=True, right_index=True, how='outer')
            node_lut.data = node_lut.data.sort_values(by="name_id")

            node_lut.save_to_pickle(preprocessed_file)
        return node_lut

    def load_relation_lut(self):
        preprocessed_file = os.path.join(self.processed_data_path, "relation_lut.pkl")
        if os.path.exists(preprocessed_file):
            relation_lut = LookUpTable()
            relation_lut.read_from_pickle(preprocessed_file)
        else:
            relation_lut = self._load_lut(self.relation_lut_name)
            relation_lut.add_vocab(self.relation_vocab)

            df = pd.DataFrame([self.relation_vocab.word2idx]).T
            df = df.rename({0: "name_id"}, axis=1)
            relation_lut.data = pd.merge(df, relation_lut.data, left_index=True, right_index=True, how='outer')
            relation_lut.data = relation_lut.data.sort_values(by="name_id")

            relation_lut.save_to_pickle(preprocessed_file)
        return relation_lut

    def load_time_lut(self):
        time_lut = LookUpTable()
        time_lut.add_vocab(self.time_vocab)
        return time_lut

    def load_all_lut(self):
        node_lut = self.load_node_lut()
        node_lut.add_processed_path(self.processed_data_path)
        relation_lut = self.load_relation_lut()
        relation_lut.add_processed_path(self.processed_data_path)
        time_lut = self.load_time_lut()
        return node_lut, relation_lut, time_lut

    def describe(self):
        tb = pt.PrettyTable()
        tb.field_names = [self.data_name, "train", "valid", "test", "node", "relation", "time"]
        tb.add_row(
            ["num", self.train_len, self.valid_len, self.test_len, len(self.node_vocab), len(self.relation_vocab),
             len(self.time_vocab)])
        print(tb)