import prettytable as pt


class Datable:
    def __init__(self):
        self.datas = {}
        self.columns = list()

    def __call__(self, head_name=None, data=None):
        if head_name == None:
            raise ValueError("Please input your column name in the first position of Datable_instance!")
        if data == None:
            raise ValueError("Please input your data in the second position of Datable_instance!")
        if isinstance(head_name, str):
            self._add_data(head_name, data)
        elif isinstance(head_name, list):
            if len(head_name) == len(data):
                self._add_datas(head_name, data)
            else:
                raise ValueError("Column and data must be corresponding!")
        else:
            raise ValueError("Name of column must be string or list!")

    def _add_data(self, head_name, data):
        if head_name not in self.columns:
            self.columns.append(head_name)
            self.datas[head_name]=list()#
            # self.datas[head_name] = data
            self.datas[head_name]= self.datas[head_name]+data
        if head_name in self.columns:
            self.datas[head_name]= self.datas[head_name]+data
            # self.datas[head_name] = data

    def _add_datas(self, head_name, data):
        count = 0
        for name in head_name:
            if name not in self.columns:
                self.columns.append(name)
                self.datas[name] = data[count]
                count = count + 1
            else:
                self.datas[name] = data[count]
                count = count + 1

    def __len__(self):
        length_list = list()
        for column in self.columns:
            length_list.append(len(self.datas[column]))
        length_set = set(length_list)
        if len(length_set) == 1:
            return length_list[0]
        else:
            raise ValueError("The numbers of elements in all columns are different!")

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_row(index)
        elif isinstance(index, str):
            return self._get_column(index)
        else:
            raise ValueError("Index must be number or string!")

    def _get_row(self, index):
        candidate = list()
        for column in self.columns:
            candidate.append(self.datas[column][index])
        return candidate

    def _get_column(self, index):
        if index in self.columns:
            return self.datas[index]
        else:
            raise ValueError("There is no corresponding column name %s!" % (index))

    def search(self,input,input_column,output_column):
        index = None
        for i,element in enumerate(self.datas[input_column]):
            if element == input:
                index=i
        if index != None:
            return self.datas[output_column][index]
        else:
            raise ValueError("Element is not in column!")

    def print_table(self):
        self._update_index_column()
        table = pt.PrettyTable(self.columns)
        max_length = self._get_max_length()
        for i in range(max_length):
            table.add_row(self._get_row(i))
        print(table)

    def _update_index_column(self):
        max_length = self._get_max_length()
        if "index" not in self.columns:
            self.columns.insert(0, "index")
            self.datas["index"]=list()
            self._add_data("index", list(range(max_length)))
        else:
            self._add_data("index", list(range(len(self.datas["index"]), max_length)))

    def _get_max_length(self):
        max = 0
        for column in self.columns:
            if len(self.datas[column]) > max:
                max = len(self.datas[column])
        return max




if __name__ == "__main__":
    data = [["小A", "小B", "小C"],
            ["在", "去", "在"],
            ["家", "超市", "学校"]]

    # __init__
    datable = Datable()                                                       # 标准实例化
    # __call__
    # datable("head",data[0])                                                 # 标准添加数据方式一
    # datable("relation",data[1])                                             # 标准添加数据方式一
    # datable("tail",data[2])                                                 # 标准添加数据方式一
    datable(["head", "relation", "tail"], [data[0], data[1], data[2]])        # 标准添加数据方式二
    # datable()                                                               # 错误添加数据方式一
    # datable("tail")                                                         # 错误添加数据方式二
    # datable(123,data[0])                                                    # 错误添加数据方式三
    # datable(["head","relation","tail"],[data[0],data[1]])                   # 标准添加数据方式四
    # __len__
    print("The length of datable is:", len(datable))                          # 标准获取长度方式一
    # __getitem__
    print("Search by row index:", datable[1])                                 # 标准索引查询行列方式一
    print("Search by columw name:", datable["tail"])                          # 标准索引查询行列方式二
    # print(datable[[1,2]])                                                   # 错误索引查询行列方式一
    # print(datable["aaaaa"])                                                 # 错误索引查询方式二
    # search
    print("Cross column search:",datable.search("小B","head","tail"))          # 标准跨类查询方式一
    # print(datable.search("aaaa","head","tail"))                             # 错误跨类查询方式一
    # print_table
    datable.print_table()                                                     # 标准打印datable方式
    datable.print_table()
