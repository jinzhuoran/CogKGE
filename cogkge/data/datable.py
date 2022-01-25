import pandas as pd
import prettytable as pt


class Datable:
    def __init__(self, data_type):
        self.data = None
        self.data_type = data_type

    def read_csv(self, file, **args):
        """
        create datable from csv file
        :param args: args for pandas read_csv function
        """
        self.data = pd.read_csv(filepath_or_buffer=file, **args)

    def __len__(self):
        return self.data.shape[0]

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

    # def __str__(self):
    #     print(self.data)

    def save_to_pickle(self, *args, **kwargs):
        self.data.to_pickle(*args, **kwargs)
        # pd.to_pickle(self.data,*args,**kwargs)

    def read_from_pickle(self, *args, **kwargs):
        self.data = pd.read_pickle(*args, **kwargs)

    def to_numpy(self):
        return self.data.values

    def str2idx(self, column_name, vocab):
        """
        convert one column of datable from str-type to idx-type
        :param column_name: which column shall be processed
        :param vocab: choose the mapping from str to idx
        """
        self.data[column_name] = self._series2numpy(self.data[column_name], vocab)

    def describe(self, front=3):
        tb = pt.PrettyTable()
        df_list = self.data.iloc[:front].values.tolist()
        if len(df_list[0]) == 5:
            tb.field_names = ["head",
                              "relation",
                              "tail",
                              "start_time",
                              "end_time"]
        if len(df_list[0]) == 3:
            tb.field_names = ["head",
                              "relation",
                              "tail"]
        for i in range(len(df_list)):
            tb.add_row(df_list[i])
        print("Show front {} lines of {}_data".format(front, self.data_type))
        print(tb)

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

    def search(self, name, attribute):
        """
        find attribute of the given name
        :param name: str
        :param attribute: str
        :return: the data stored in the corresponding place
        """
        return self.data.loc[name, attribute]