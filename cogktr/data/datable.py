import prettytable as pt
import pandas as pd
from .vocabulary import Vocabulary


class Datable:
    def __init__(self):
        self.data = None

    def read_csv(self,file, **args):
        """
        create datable from csv file
        :param args: args for pandas read_csv function
        """
        self.data = pd.read_csv(filepath_or_buffer=file,**args)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_row(index)
        elif isinstance(index, str):
            return self._get_column(index)
        else:
            raise ValueError("Index must be number or string!")

    def _get_row(self,index):
        return self.data.iloc[index,:]

    def _get_column(self,index):
        """
        get the column of dataframe
        :param index: column name
        :return: pandas Series
        """
        return self.data[index]

    # def __str__(self):
    #     print(self.data)

    def save_to_pickle(self,*args,**kwargs):
        self.data.to_pickle(*args,**kwargs)
        # pd.to_pickle(self.data,*args,**kwargs)

    def read_from_pickle(self,*args,**kwargs):
        self.data = pd.read_pickle(*args,**kwargs)

    def to_numpy(self):
        return self.data.values

    def str2idx(self,column_name,vocab):
        """
        convert one column of datable from str-type to idx-type
        :param column_name: which column shall be processed
        :param vocab: choose the mapping from str to idx
        """
        self.data[column_name] = self._series2numpy(self.data[column_name], vocab)

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


# class Datable:
#     def __init__(self):
#         self.datas = {}
#         self.columns = list()
#
#     def __call__(self, head_name=None, data=None):
#         if head_name == None:
#             raise ValueError("Please input your column name in the first position of Datable_instance!")
#         if data == None:
#             raise ValueError("Please input your data in the second position of Datable_instance!")
#         if isinstance(head_name, str):
#             self._add_data(head_name, data)
#         elif isinstance(head_name, list):
#             if len(head_name) == len(data):
#                 self._add_datas(head_name, data)
#             else:
#                 raise ValueError("Column and data must be corresponding!")
#         else:
#             raise ValueError("Name of column must be string or list!")
#
#     def _add_data(self, head_name, data):
#         if head_name not in self.columns:
#             self.columns.append(head_name)
#             self.datas[head_name] = list()  #
#             # self.datas[head_name] = data
#             self.datas[head_name] = self.datas[head_name] + data
#         if head_name in self.columns:
#             self.datas[head_name] = self.datas[head_name] + data
#             # self.datas[head_name] = data
#
#     def _add_datas(self, head_name, data):
#         count = 0
#         for name in head_name:
#             if name not in self.columns:
#                 self.columns.append(name)
#                 self.datas[name] = data[count]
#                 count = count + 1
#             else:
#                 self.datas[name] = data[count]
#                 count = count + 1
#
#     def __len__(self):
#         length_list = list()
#         for column in self.columns:
#             length_list.append(len(self.datas[column]))
#         length_set = set(length_list)
#         if len(length_set) == 1:
#             return length_list[0]
#         else:
#             raise ValueError("The numbers of elements in all columns are different!")
#
#     def __getitem__(self, index):
#         if isinstance(index, int):
#             return self._get_row(index)
#         elif isinstance(index, str):
#             return self._get_column(index)
#         else:
#             raise ValueError("Index must be number or string!")
#
#     def _get_row(self, index, visua_length=None):
#         if visua_length == None:
#             candidate = list()
#             for column in self.columns:
#                 candidate.append(self.datas[column][index])
#             return candidate
#         else:
#             candidate = list()
#             for column in self.columns:
#                 if isinstance(self.datas[column][index], str) and len(self.datas[column][index]) > visua_length:
#                     candidate.append(
#                         self.datas[column][index][:visua_length] + "..." + "(Show the first %s words)" % (visua_length))
#                 else:
#                     candidate.append(self.datas[column][index])
#             return candidate
#
#     def _get_column(self, index):
#         if index in self.columns:
#             return self.datas[index]
#         else:
#             raise ValueError("There is no corresponding column name %s!" % (index))
#
#     def search(self, input, input_column, output_column):
#         index = None
#         for i, element in enumerate(self.datas[input_column]):
#             if element == input:
#                 index = i
#         if index != None:
#             return self.datas[output_column][index]
#         else:
#             raise ValueError("Element is not in column!")
#
#     def print_table(self, num=None):
#         self._update_index_column()
#         table = pt.PrettyTable(self.columns)
#         if num == None:
#             max_length = self._get_max_length()
#         else:
#             max_length = num
#         for i in range(max_length):
#             table.add_row(self._get_row(i, visua_length=40))
#         print(table)
#
#     def _update_index_column(self):
#         max_length = self._get_max_length()
#         if "index" not in self.columns:
#             self.columns.insert(0, "index")
#             self.datas["index"] = list()
#             self._add_data("index", list(range(max_length)))
#         else:
#             self._add_data("index", list(range(len(self.datas["index"]), max_length)))
#
#     def _get_max_length(self):
#         max = 0
#         for column in self.columns:
#             if len(self.datas[column]) > max:
#                 max = len(self.datas[column])
#         return max
#
#
# if __name__ == "__main__":
#     data = [["小A", "小B", "小C"],
#             ["在", "去", "在"],
#             ["家", "超市", "学校"]]
#
#     # __init__
#     datable = Datable()  # 标准实例化
#     # __call__
#     # datable("head",data[0])                                                 # 标准添加数据方式一
#     # datable("relation",data[1])                                             # 标准添加数据方式一
#     # datable("tail",data[2])                                                 # 标准添加数据方式一
#     datable(["head", "relation", "tail"], [data[0], data[1], data[2]])  # 标准添加数据方式二
#     # datable()                                                               # 错误添加数据方式一
#     # datable("tail")                                                         # 错误添加数据方式二
#     # datable(123,data[0])                                                    # 错误添加数据方式三
#     # datable(["head","relation","tail"],[data[0],data[1]])                   # 标准添加数据方式四
#     # __len__
#     print("The length of datable is:", len(datable))  # 标准获取长度方式一
#     # __getitem__
#     print("Search by row index:", datable[1])  # 标准索引查询行列方式一
#     print("Search by columw name:", datable["tail"])  # 标准索引查询行列方式二
#     # print(datable[[1,2]])                                                   # 错误索引查询行列方式一
#     # print(datable["aaaaa"])                                                 # 错误索引查询方式二
#     # search
#     print("Cross column search:", datable.search("小B", "head", "tail"))  # 标准跨类查询方式一
#     # print(datable.search("aaaa","head","tail"))                             # 错误跨类查询方式一
#     # print_table
#     datable.print_table()  # 标准打印datable方式
#     datable.print_table(2)
