import pickle

import pandas as pd


class LookUpTable:
    def __init__(self):
        self.data = None
        self.vocab = None
        self.token = None
        self.mask = None
        self.type = None
        self.processed_path = None

    def add_vocab(self, vocab):
        self.vocab = vocab

    def add_token(self, token):
        self.token = token

    def add_mask(self, mask):
        self.mask = mask

    def add_type(self, type):
        self.type = type

    def add_processed_path(self, processed_path):
        self.processed_path = processed_path

    def read_json(self, *args, **kwargs):
        self.data = pd.read_json(*args, **kwargs)

    def read_csv(self, *args, **kwargs):
        self.data = pd.read_csv(*args, **kwargs)

    def transpose(self):
        self.data = self.data.T

    def __len__(self):
        return len(self.vocab)

    def save_to_pickle(self, file_name):
        tmp = {"data": self.data, "vocab": self.vocab, "token": self.token, "mask": self.mask, "type": self.type,
               "processed_path": self.processed_path}
        with open(file_name, "wb") as f:
            pickle.dump(tmp, f, protocol=pickle.HIGHEST_PROTOCOL)

    def read_from_pickle(self, file_name):
        with open(file_name, "rb") as f:
            tmp = pickle.load(f)
            self.data = tmp["data"]
            self.vocab = tmp["vocab"]
            self.token = tmp["token"]
            self.mask = tmp["mask"]
            self.type = tmp["type"]
            self.processed_path = tmp["processed_path"]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_row(index)
        elif isinstance(index, str):
            return self._get_column(index)
        else:
            raise ValueError("Index must be number or string!")

    def _get_row(self, index):
        return self.data.iloc[index, :]

    def _get_column(self, index):
        """
        get the column of dataframe
        :param index: column name
        :return: pandas Series
        """
        return self.data[index]

    def rename(self, *args, **kwargs):
        self.data = self.data.rename(*args, **kwargs)

    def add_column(self, elem_list, column_name):
        """
        add column to current dataframe
        :param column: a list of elements the same as the dataframe length
        :return: nothing
        """
        kwargs = {column_name: pd.Series(elem_list).values}
        self.data = self.data.assign(**kwargs)

    def append(self, lut2):
        new_lut = LookUpTable()
        new_lut.data = self.data.append(lut2.data)
        return new_lut

    def search(self, name, attribute):
        """
        find attribute of the given name
        :param name: str
        :param attribute: str
        :return: the data stored in the corresponding place
        """
        return self.data.loc[name, attribute]

    def describe(self, front=3, max_colwidth=100):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('max_colwidth', max_colwidth)
        print(self.data.iloc[:front])
