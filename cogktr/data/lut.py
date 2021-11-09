####################################！！！！！！！最开始版本 ！！！##########################################################
# class LUT:
#     def __init__(self,entity2id,relation2id):
#         """
#         entity2id:entity names -> entity id  dictionary
#         relation2id:relation names -> relation id   dictionary
#         """
#         self.entity2id_ = entity2id
#         self.relation2id_= relation2id
#         self.id2entity_ = {value:key for key,value in entity2id.items()}
#         self.id2relation_ = {value:key for key,value in relation2id.items()}
#
#     def entity2id(self,entity):
#         return self.entity2id_[entity]
#
#     def id2entity(self,id):
#         return self.id2entity_[id]
#
#     def relation2id(self,relation):
#         return self.relation2id_[relation]
#
#     def id2relation(self,id):
#         return self.id2relation_[id]
#
#     def num_entity(self):
#         return len(self.entity2id_)
#
#     def num_relation(self):
#         return len(self.relation2id_)
#
#
######################################## ！！！！pandas版本！！！！！########################################################
# import pandas as pd
#
#
# class LookUpTable:
#     def __init__(self,
#                  type=None,
#                  nodename=None,        #最关键，必须输入一个实体的列表
#                  nodealiases=None,
#                  typename=None,
#                  text=None,
#                  ):
#         pd.set_option('display.max_columns', None)
#         pd.set_option('expand_frame_repr', False)
#
#         self.type=type
#         self.nodename=nodename
#         self.nodealiases=nodealiases
#         self.typename=typename
#         self.text=text
#         self.embedding=None
#
#         self.nodename_frame=None
#         self.nodealiases_frame=None
#         self.typename_frame=None
#         self.text_frame=None
#         self.embedding_frame=None
#
#         self.table=None
#         self._load_data_to_table()
#         self._merge_table()
#         self.sort()
#
#     def _load_data_to_table(self):
#         if self.nodename!=None:
#             self.nodename_frame=pd.DataFrame({"E_or_R":self.type,
#                                                 "node_name":self.nodename})
#         else:
#             raise ValueError("Entity/Relation list should not be empty!")
#         if self.nodealiases!=None:
#             self.nodealiases_frame=pd.DataFrame({"node_name":self.nodealiases[0],
#                                                    "node_aliases":self.nodealiases[1]})
#         else:
#             self.nodealiases_frame=pd.DataFrame({"node_name":self.nodename,
#                                                    "node_aliases":None})
#         if self.typename!=None:
#             self.typename_frame=pd.DataFrame({"node_name":self.typename[0],
#                                                 "type_name":self.typename[1]})
#         else:
#             self.typename_frame=pd.DataFrame({"node_name":self.nodename,
#                                                 "type_name":None})
#         #self.typeindex==None:
#         if self.typename != None:
#             #去重并防止变顺序
#             typename_set=list(set(self.typename[1]))
#             typename_set.sort(key=self.typename[1].index)
#             self.typeindex_frame = pd.DataFrame({"type_name": typename_set,
#                                                    "type_index": list(range(len(list(typename_set))))})
#         else:
#             self.typeindex_frame = pd.DataFrame({"node_name": self.nodename,
#                                                    "type_index": None})
#
#         if self.text!=None:
#             self.text_frame = pd.DataFrame({"node_name": self.text[0],
#                                               "text": self.text[1]})
#         else:
#             self.text_frame = pd.DataFrame({"node_name": self.nodename,
#                                               "text": None})
#         if self.embedding==None:
#             self.embedding_frame = pd.DataFrame({"node_name": self.nodename,
#                                                    "embedding": None})
#
#
#
#     def _merge_table(self):
#         self.table=pd.merge(self.nodename_frame,self.nodealiases_frame,on=["node_name"],how="outer")
#         self.table=pd.merge(self.table, self.typename_frame, on=["node_name"], how="outer")
#         if self.typename != None:
#             self.table = pd.merge(self.table, self.typeindex_frame, on=["type_name"], how="outer")
#         else:
#             self.table = pd.merge(self.table, self.typeindex_frame, on=["node_name"], how="outer")
#         self.table = pd.merge(self.table, self.text_frame, on=["node_name"], how="outer")
#         self.table = pd.merge(self.table, self.embedding_frame, on=["node_name"], how="outer")
#
#     def sort(self):
#         self.table=self.table.sort_index(axis=0)
#
#
#
# if __name__ == "__main__":
#     train_data=[["E_1","E_2","E_3"],     #head
#                 ["R_1","R_2","R_3"],     #relation
#                 ["E_4","E_5","E_1"]]     #tail
#     entity_aliases=[["E_1","E_2","E_3","E_4","E_5"],
#                     [["cat","Cat","CAT"],["woman"],["monkey","monkeys"],["fish","Fish"],["man"]]]
#     relation_aliases=[["R_1","R_2","R_3"],
#                       [["Aa","AA","aA"],["bb","Bb"],["c"]]]
#     entity_type=[["E_1","E_2","E_3","E_4","E_5"],
#                  ["animal","human","animal","animal","human"]]
#
#     relation_type =[["R_1","R_2","R_3"],
#                     ["double","double","single"]]
#     entity_text =[["E_1","E_2","E_3","E_4","E_5"],
#                   ["cat is........................",
#                    "woman is......................",
#                    "monkey is.....................",
#                    "fish is.......................",
#                    "man is........................"]]
#
#     # print("train_data:\n",train_data)
#     # print("entity_aliases:\n",entity_aliases)
#     # print("realtion_aliases:\n",relation_aliases)
#     # print("entity_type:\n",entity_type)
#     # print("relation_type:\n",relation_type)
#     # print("entity_text:\n",entity_text)
#                     name          aliases               type        typeindex       text        embedding
#     ########################################################################################################
#     #  E_or_R  #  node_name  #     node_aliases    #  type_name  #  type_index  #    text    #  embedding  #
#     ########################################################################################################
#     #   "E"    #    "E_1"    # ["cat","Cat","CAT"] #   "animal"  #      0       # "cat......"#[,,,,,,,,,,,]#
#     #..........#.............#.....................#.............#..............#............#.............#
#     #..........#.............#.....................#.............#..............#............#.............#
#     ########################################################################################################
#     #  E_or_R  #  node_name  #     node_aliases    #  type_name  #  type_index  #    text    #  embedding  #
#     ########################################################################################################
#     #   "R"    #    "R_1"    #  ["Aa","AA","aA"]   #   "double"  #      0       #    None    #[,,,,,,,,,,,]#
#     #..........#.............#.....................#.............#..............#............#.............#
#     #..........#.............#.....................#.............#..............#............#.............#
#     ########################################################################################################
#     #注释：建立的时候全部以node_name为枢纽一一对应                                                                #
#     ########################################################################################################
#     lookuptable_E=LookUpTable(type="E",
#                               nodename=entity_aliases[0],      #必填
#                               nodealiases=entity_aliases,      #选填
#                               typename=entity_type,            #选填
#                               text=entity_text)                #选填
#     #查整个表
#     print("lookuptable_E.table:-------------------->\n",lookuptable_E.table)
#     #查索引1的所有信息
#     print("lookuptable_E.table.loc[1]:-------------------->\n",lookuptable_E.table.loc[1])
#     # 查索引1的"node_aliases"
#     print("lookuptable_E.table.loc[1,\"node_aliases\"]:-------------------->\n",lookuptable_E.table.loc[1,"node_aliases"])
#     lookuptable_E.table.to_csv("lookuptable.csv")
#     # lookuptable_R = LookUpTable(type="R",
#     #                             nodename=relation_aliases[0],  # 必填
#     #                             nodealiases=relation_aliases,  # 选填
#     #                             typename=relation_type,  # 选填
#     #                             text=None)  # 选填
#     # print(lookuptable_R.table)
###############################################！！！！自己的数据结构版本 ！！！################################################
import prettytable as pt

class LookUpTable:
    def __init__(self):
        self.columns=list()
        self.datas=dict()

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
        for i in range(len(mydict.keys())):
            mylist.append(mydict[i])
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
            self._add_data("index", list(range(max_length)))
        else:
            self._add_data("index", list(range(len(self.datas["index"]), max_length)))

    def _get_max_length(self):
        max = 0
        for column in self.columns:
            if len(self.datas[column]) > max:
                max = len(self.datas[column])
        return max

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

    lookuptable_E=LookUpTable()
    lookuptable_E.create_table(create_dic=True,item_list=entity_name_list)
    lookuptable_E("aliases",entity_aliases)
    lookuptable_E("type", entity_type)
    lookuptable_E("text", entity_text)
    lookuptable_E.print_table()
    print(len(lookuptable_E))
    print(lookuptable_E[3])
    print(lookuptable_E["aliases"])
    print(lookuptable_E["text"][4])

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











