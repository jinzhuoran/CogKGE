import prettytable as pt

class LookUpTable:
    def __init__(self):
        self.columns=list()
        self.datas=dict()
        self.str_dic=None

    def create_table(self,create_dic,str_dic=None,item_list=None):
        self.create_dic = create_dic
        self.str_dic = str_dic
        self.item_list = item_list
        if self.create_dic == True:
            if self.item_list != None:
                self.str_dic = dict()
                for index, str in enumerate(item_list):
                    self.str_dic[str] = index
            else:
                raise ValueError("Please enter item_list!")
        if self.create_dic == False:
            if self.str_dic != None:
                self.str_dic = self.str_dic
            else:
                raise ValueError("Please enter str_dict!")
        name_dict = dict()
        for i in list(self.str_dic.keys()):
            name_dict[i] = i
        name_list = self._dict2list(name_dict)
        self._add_data("name", name_list)

    def __call__(self,head_name=None,data_dict=None):
        data_list=self._dict2list(data_dict)
        self._add_data(head_name,data_list)

    def _dict2list(self,mydict):
        mylist=list()
        key_list=list(mydict.keys())
        for i in key_list:
            mydict[self.str_dic[i]]=mydict.pop(i)
        for i in range(len(self.str_dic)):
            try:
                mylist.append(mydict[i])
            except Exception as e:
                mylist.append(None)
        return mylist

    def _add_data(self, head_name, data):
        if head_name not in self.columns:
            self.columns.append(head_name)
            self.datas[head_name] = data
        if head_name in self.columns:
            self.datas[head_name] = data

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

    def _getitem(self, index):
        if isinstance(index, int):
            return self._get_row(index)
        elif isinstance(index, str):
            return self._get_column(index)
        else:
            raise ValueError("Index must be number or string!")

    def _get_row(self, index,visua_descriptions_length=None):
        # candidate = list()
        # for column in self.columns:
        #     candidate.append(self.datas[column][index])
        # return candidate
        if visua_descriptions_length==None:
            candidate = list()
            for column in self.columns:
                candidate.append(self.datas[column][index])
            return candidate
        else:
            candidate = list()
            for column in self.columns:
                if column=="descriptions":
                    candidate.append(self.datas[column][index][:visua_descriptions_length]+"..."+"(Show the first %s words)"%(visua_descriptions_length))
                else:
                    candidate.append(self.datas[column][index])
            return candidate

    def _get_column(self, index):
        if index in self.columns:
            return self.datas[index]
        else:
            raise ValueError("There is no corresponding column name %s!" % (index))

    def print_table(self,num=None):
        self._update_index_column()
        table = pt.PrettyTable(self.columns)
        if num==None:
            max_length = self._get_max_length()
        else:
            max_length=num
        for i in range(max_length):
            table.add_row(self._get_row(i,visua_descriptions_length=80))
        print(table)

    def _update_index_column(self):
        max_length = self._get_max_length()
        if "index" not in self.columns:
            self.columns.insert(0, "index")
            self._add_data("index", list(range(max_length)))
        else:
            self._add_data("index", list(range(max_length)))

    def _get_max_length(self):
        max = 0
        for column in self.columns:
            if len(self.datas[column]) > max:
                max = len(self.datas[column])
        return max

    def batch_index2categort(self,batch_data,category):
        category_list=list()
        for i in range(len(batch_data)):
            category_list.append(self._getitem(category)[batch_data[i]])
        return category_list

    def save_table(self):
        pass

    def load_table(self):
        pass

if __name__=="__main__":
    entity_aliases={"E_1":["cat","Cat","CAT"],
                    "E_2":["woman"],
                    "E_3":["monkey","monkeys"],
                    "E_4":["fish","Fish"],
                    "E_5":["man"]}
    entity_type={"E_1":"animal",
                 "E_2":"human",
                 "E_3":"animal",
                 "E_4":"animal",
                 "E_5":"human"}
    entity_text={"E_1":"cat is........................",
                 "E_2":"woman is......................",
                 "E_3":"monkey is.....................",
                 "E_4":"fish is.......................",
                 "E_5":"man is........................"}
    relation_aliases ={"R_1":["Aa","AA","aA"],
                       "R_2":["bb","Bb"],
                       "R_3":["c"]}
    relation_type ={"R_1":"double",
                       "R_2":"double",
                       "R_3":"single"}
    entity_name_list = list(set(list(entity_aliases.keys())))
    entity_name_list.sort(key=list(entity_aliases.keys()).index)

    print("entity_aliases:\n",entity_aliases)
    print("entity_type:\n",entity_type)
    print("entity_text:\n",entity_text)
    print("realtion_aliases:\n",relation_aliases)
    print("relation_type:\n",relation_type)

    #建立空表
    lookuptable_E=LookUpTable()
    #表的初始化
    lookuptable_E.create_table(create_dic=True,item_list=entity_name_list)
    #增加列
    lookuptable_E("aliases",entity_aliases)
    lookuptable_E("type", entity_type)
    lookuptable_E("text", entity_text)
    #打印整个表
    lookuptable_E.print_table()
    lookuptable_E.print_table(2)
    #表长
    print(len(lookuptable_E))
    #访问行
    print(lookuptable_E[3])
    #访问列
    print(lookuptable_E["aliases"])
    #同时访问行列
    print(lookuptable_E["text"][4])
    #名字转索引
    print(lookuptable_E.str_dic["E_2"])
    #批量将idx(实际为一个一维tensor)转为其他类型的数据
    import torch
    input_index=torch.tensor([2,1,3])
    output_category=lookuptable_E.batch_index2categort(batch_data=input_index,category="name")
    print(output_category)

    #以下是LookUpTable规划的完整形式
    #     ########################################################################################################
    #     #  index   #     name    #      aliases        #    type     #  typeindex   #    text    #  embedding  #
    #     ########################################################################################################
    #     #    0     #    "E_1"    # ["cat","Cat","CAT"] #   "animal"  #      0       # "cat......"#[,,,,,,,,,,,]#
    #     #..........#.............#.....................#.............#..............#............#.............#
    #     #..........#.............#.....................#.............#..............#............#.............#
    #     ########################################################################################################
    #     #  index   #  node_name  #     node_aliases    #  type_name  #  type_index  #    text    #  embedding  #
    #     ########################################################################################################
    #     #    0     #    "R_1"    #  ["Aa","AA","aA"]   #   "double"  #      0       #    None    #[,,,,,,,,,,,]#
    #     #..........#.............#.....................#.............#..............#............#.............#
    #     #..........#.............#.....................#.............#..............#............#.............#
    #     ########################################################################################################











