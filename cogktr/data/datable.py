import pandas as pd
import prettytable as pt
class Datable_1:
    def __init__(self,heads=None):
        self.heads=heads
        self.table=None
        self.table = pd.DataFrame(columns=["head",
                                           "relation",
                                           "tail"])
    def load_data_to_table(self,data):
        self.table["head"]=  data[0]
        self.table["relation"]=  data[1]
        self.table["tail"]=data[2]
        return self.table
class Datable_2:
    def __init__(self):
        self.datas={}
        self.columns=list()
    def __call__(self,head_name=None,data=None):
        if isinstance(head_name,str):
            self.add_data(head_name,data)
        if isinstance(head_name,list):
            self.add_datas(head_name,data)
    def add_data(self,head_name,data):
        if head_name not in self.columns:
            self.columns.append(head_name)
            self.datas[head_name]=data
        if head_name in self.columns:
            self.datas[head_name] = data
    def add_datas(self,head_name,data):
        count=0
        for name in head_name:
            if name not in self.columns:
                self.columns.append(name)
                self.datas[name] = data[count]
                count=count+1
            if name in self.columns:
                self.datas[name] = data[count]
                count = count + 1
    def __len__(self):
        return len(self.datas["head"])
    def __getitem__(self,index):
        if isinstance(index,int):
            return self.get_row(index)
        if isinstance(index,str):
            return self.get_column(index)
    def get_row(self,index):
        candidate=list()
        for column in self.columns:
            candidate.append(self.datas[column][index])
        return candidate
    def get_column(self,index):
        return self.datas[index]
    def print_table(self):
        self.add_index_column()
        table=pt.PrettyTable(self.columns)
        max_length=self.get_max_length()
        for i in range(max_length):
            table.add_row(self.get_row(i))
        print(table)
    def get_max_length(self):
        max=0
        for column in self.columns:
            if len(self.datas[column])>max:
                max=len(self.datas[column])
        return max
    def add_index_column(self):
        max_length=self.get_max_length()
        self.columns.insert(0, "index")
        self.add_data("index",list(range(max_length)))








if __name__ == "__main__":
    data=[["小A","小B","小C"],
          ["在","去","在"],
          ["家","超市","学校"]]


    datable_1= Datable_1(heads=None)
    datable_1= datable_1.load_data_to_table(data)
    print("以下是调用pandas写的datable\n",datable_1)



    datable_2=Datable_2()
    datable_2("head",data[0])
    datable_2("relation",data[1])
    datable_2("tail",data[2])
    print("以下是咱们自己写的datable\n")
    datable_2.print_table()
    # print(datable_2[2])
    # print(datable_2["head"])