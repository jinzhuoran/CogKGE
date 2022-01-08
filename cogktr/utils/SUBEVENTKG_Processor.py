import os
import re
import math
import mmap
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from sklearn.utils import shuffle


class SUBEVENTKG_Processor(object):
    """
    对EVENTKG数据集取适用于知识表示任务的子数据集
    原数据集地址链接在https://eventkg.l3s.uni-hannover.de/data.html，采用3.0版本的数据包
    """
    def __init__(self,
                 events_nq_path,
                 entities_nq_path,
                 relations_base_nq_path,
                 relations_events_other_nq_path,
                 relations_entities_temporal_nq_path,
                 processed_entities_path,
                 processed_events_path,
                 processed_event_event_path,
                 processed_event_entity_path,
                 processed_entity_entity_path,
                 filt_event_event_path,
                 filt_event_entity_path,
                 filt_entity_entity_path,
                 event_node_count_path,
                 entity_node_count_path,
                 event_rdf2name_path,
                 entity_rdf2name_path,
                 relation_rdf2name_path,
                 event_lut_path,
                 entity_lut_path,
                 relation_lut_path,
                 event_degree_list,
                 entity_degree_list
                 ):
        """
        Args:
            events_nq_path:事件节点的原始数据集
            entities_nq_path:实体节点的原始数据集
            relations_base_nq_path:事件-事件（无时间信息）原始数据集
            relations_events_other_nq_path:事件-实体，实体-事件原始数据集
            relations_entities_temporal_nq_path:实体-实体(有时间信息)原始数据集
            processed_entities_path:实体的字典,rdf->count(0)
            processed_events_path:事件的字典,rdf->count(0)
            processed_event_event_path:转化为五元组格式的 事件-事件
            processed_event_entity_path:转化为五元组格式的 事件-实体
            processed_entity_entity_path:转化为五元组格式的 实体-实体
            filt_event_event_path:过滤后的 事件-事件 五元组
            filt_event_entity_path:过滤后的 事件-实体 五元组
            filt_entity_entity_path:过滤后的 实体-实体 五元组
            event_node_count_path:统计出来的事件节点个数
            entity_node_count_path:统计出来的实体节点个数
            event_rdf2name_path:事件rdf转name
            entity_rdf2name_path:实体rdf转name
            relation_rdf2name_path:关系rdfname
            event_lut_path:事件查找表路径
            entity_lut_path:实体查找表路径
            relation_lut_path:关系查找表路径
            event_degree_list:事件度的列表
            entity_degree_list:实体度的列表
        """
        self.raw_events_path=events_nq_path
        self.raw_entities_path=entities_nq_path
        self.raw_event_event_path=relations_base_nq_path
        self.raw_event_entity_path=relations_events_other_nq_path
        self.raw_entity_entity_path=relations_entities_temporal_nq_path
        self.processed_entities_path=processed_entities_path
        self.processed_events_path=processed_events_path
        self.processed_event_event_path=processed_event_event_path
        self.processed_event_entity_path=processed_event_entity_path
        self.processed_entity_entity_path=processed_entity_entity_path
        self.filt_event_event_path=filt_event_event_path
        self.filt_event_entity_path=filt_event_entity_path
        self.filt_entity_entity_path=filt_entity_entity_path
        self.event_node_count_path=event_node_count_path
        self.entity_node_count_path=entity_node_count_path
        self.event_rdf2name_path=event_rdf2name_path
        self.entity_rdf2name_path=entity_rdf2name_path
        self.relation_rdf2name_path=relation_rdf2name_path
        self.event_lut_path=event_lut_path
        self.entity_lut_path=entity_lut_path
        self.relation_lut_path=relation_lut_path
        self.event_degree_list=event_degree_list
        self.entity_degree_list=entity_degree_list

        self.entity_dict=None
        self.event_dict=None
        self.rdf_triplets_event_event=None
        self.rdf_triplets_event_entity=None
        self.rdf_triplets_entity_entity = None
        self.filt_triplets_event_event=None
        self.filt_triplets_event_entity=None
        self.filt_triplets_entity_entity = None
        self.event_rdf2name_dict=dict()
        self.entity_rdf2name_dict=dict()
        self.relation_rdf2name_dict=dict()


    def _get_num_lines(self,file_path):
        """
        统计txt文件的行数

        :param file_path:待统计行数的txt文件路径
        """
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def create_entities_index(self,reprocess=True,describe=True):
        """
        建立{实体rdf：count}字典，这个实体是全集
        Args:
            reprocess:True为重新处理
            describe:True为显示数据集信息
        """
        if reprocess:
            self.entity_dict=dict()
            print("processing entities index...")
            with open(self.raw_entities_path, "r", encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines(self.raw_entities_path)):
                    line = line.strip().split(" ")
                    entity=line[0]
                    if entity not in self.entity_dict.keys():
                        self.entity_dict[entity]=0
            json.dump(self.entity_dict, open(self.processed_entities_path, "w"), indent=4, sort_keys=True)
            print("processed_entities_dict has been saved in {}".format(self.processed_entities_path))

        else:
            if os.path.exists(self.processed_entities_path):
                print("loading entities index...")
                with open (self.processed_entities_path) as file:
                    self.entity_dict = json.load(file)
                print("loading entities index succeed!")
            else:
                raise FileNotFoundError("processed_entities_path does not exists!")

        if describe:
            print("entities_dict_len",len(self.entity_dict))


    def create_events_index(self,reprocess=True,describe=True):
        """
        建立{事件rdf：count}字典，这个事件是全集
        """
        if reprocess:
            self.event_dict = dict()
            print("processing events index...")
            with open(self.raw_events_path, "r", encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines(self.raw_events_path)):
                    line = line.strip().split(" ")
                    event = line[0]
                    if event not in self.event_dict.keys():
                        self.event_dict[event] = 0
            json.dump(self.event_dict, open(self.processed_events_path, "w"), indent=4, sort_keys=True)
            print("processed_events_dict has been saved in {}".format(self.processed_events_path))
        else:
            if os.path.exists(self.processed_events_path):
                print("loading events index...")
                with open(self.processed_events_path) as file:
                    self.event_dict = json.load(file)
                print("loading events index succeed!")
            else:
                raise FileNotFoundError("processed_entities_path does not exists!")

        if describe:
            print("events_dict_len",len(self.event_dict))

    def event_event_raw2df(self,reprocess=True,describe=True):
        """
        找出事件与事件的hassubevent，nextevent，previousevent三种关系，并转化成dataframe格式保存

        原始格式
        事件 关系 事件
        存储格式
        事件 关系 事件 开始时间 结束时间 （事件和事件三元组没有时间信息，表示为-1）
        """
        if reprocess:
            df_lines=[]
            with open(self.raw_event_event_path, "r", encoding="utf-8") as file:
                print("processing event_event_raw2df...")
                for line in tqdm(file, total=self._get_num_lines(self.raw_event_event_path)):
                    line = line.strip().split(" ")
                    if line[1] == "<http://dbpedia.org/ontology/nextEvent>" or \
                            line[1] == "<http://dbpedia.org/ontology/previousEvent>" or \
                            line[1] == "<http://semanticweb.cs.vu.nl/2009/11/sem/hasSubEvent>":
                        head = line[0]
                        relation = line[1]
                        tail = line[2]
                        df_lines.append([head,relation,tail,-1,-1])
            self.rdf_triplets_event_event=pd.DataFrame(df_lines)
            self.rdf_triplets_event_event.columns=["head","relation","tail","start_time","end_time"]
            self.rdf_triplets_event_event.to_csv(self.processed_event_event_path)
            print("rdf_triplets_event_event has been saved in {}".format(self.processed_event_event_path))
        else:
            if os.path.exists(self.processed_event_event_path):
                print("loading event_event_raw2df...")
                self.rdf_triplets_event_event=pd.read_csv(self.processed_event_event_path)
                print("loading event_event_raw2df succeed!")
            else:
                raise FileNotFoundError("processed_event_event_path does not exists!")
        if describe:
            print("rdf_triplets_event_event_len",len(self.rdf_triplets_event_event))

    def _node_relation_datatype_raw2df(self,
                                       reprocess=True,
                                       describe=True,
                                       datatype=None,
                                       raw_data_path=None,
                                       saved_path=None):
        def init_relation_node_dict(relation_node_dict,raw_data_path):
            """嵌套字典初始化"""
            with open(raw_data_path, "r", encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines(raw_data_path)):
                    line = line.strip().split(" ")
                    relation_node = line[0]
                    if relation_node not in relation_node_dict.keys():
                        relation_node_dict[relation_node]["head"] = -1
                        relation_node_dict[relation_node]["relation"] = -1
                        relation_node_dict[relation_node]["tail"] = -1
                        relation_node_dict[relation_node]["start_time"] = -1
                        relation_node_dict[relation_node]["end_time"] = -1
            return relation_node_dict

        def add_value_relation_node_dict(relation_node_dict,raw_data_path):
            """嵌套字典添加值"""
            with open(raw_data_path, "r", encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines(raw_data_path)):
                    line = line.strip().split(" ")
                    relation_node = line[0]
                    arrow = line[1]
                    value = line[2]
                    if arrow == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>":
                        relation_node_dict[relation_node]["head"] = value
                    if arrow == "<http://semanticweb.cs.vu.nl/2009/11/sem/roleType>":
                        relation_node_dict[relation_node]["relation"] = value
                    if arrow == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>":
                        relation_node_dict[relation_node]["tail"] = value
                    if arrow == "<http://semanticweb.cs.vu.nl/2009/11/sem/hasBeginTimeStamp>":
                        relation_node_dict[relation_node]["start_time"] = value
                    if arrow == "<http://semanticweb.cs.vu.nl/2009/11/sem/hasEndTimeStamp>":
                        relation_node_dict[relation_node]["end_time"] = value
            return relation_node_dict

        if reprocess:
            relation_node_dict = defaultdict(dict)
            print("processing {} _raw2df...".format(datatype))
            relation_node_dict = init_relation_node_dict(relation_node_dict,raw_data_path)
            relation_node_dict = add_value_relation_node_dict(relation_node_dict,raw_data_path)
            # 嵌套字典转dataframe
            df_lines = []
            for key in tqdm(relation_node_dict.keys()):
                df_lines.append([relation_node_dict[key]["head"],
                                 relation_node_dict[key]["relation"],
                                 relation_node_dict[key]["tail"],
                                 relation_node_dict[key]["start_time"],
                                 relation_node_dict[key]["end_time"]])
            df = pd.DataFrame(df_lines)
            df.columns = ["head", "relation", "tail", "start_time", "end_time"]
            if datatype=="event_entity":
                self.rdf_triplets_event_entity=df
            if datatype=="entity_entity":
                self.rdf_triplets_entity_entity=df
            df.to_csv(saved_path)
            print("rdf_triplets_{} has been saved in {}".format(datatype,saved_path))

        else:
            if os.path.exists(saved_path):
                print("loading {}_raw2df...".format(datatype))
                df = pd.read_csv(saved_path)
                if datatype=="event_entity":
                    self.rdf_triplets_event_entity=df
                if datatype=="entity_entity":
                    self.rdf_triplets_entity_entity=df
                print("loading {}_raw2df succeed!".format(datatype))
            else:
                raise FileNotFoundError("processed_{}_path does not exists!".format(datatype))
        if describe:
            print("rdf_triplets_{}_len".format(datatype), len(df))

    def event_entity_raw2df(self,reprocess=True,describe=True):
        """
        找出事件与实体，实体与事件的所有关系，并转化成dataframe格式保存
        原始格式--》嵌套字典格式--》存储格式dataframe

        原始格式
        relation结点 subject event/entity (头结点)
        relation结点 object event/entity (尾结点)
        relation结点 roleType 比如参与者(关系)
        relation结点 hasbegintimestamp 时间
        relation结点 hasendtimestamp 时间

        嵌套字典格式
        {relation结点：{头结点，关系，尾结点，开始时间，结束时间}。。。。。。。。。。}

        存储格式
        事件/实体 关系 实体/事件 开始时间 结束时间 如果有空值，则用-1表示，多个关系则随机选取一个
        """
        self._node_relation_datatype_raw2df(reprocess=reprocess,
                                            describe=describe,
                                            datatype="event_entity",
                                            raw_data_path=self.raw_event_entity_path,
                                            saved_path=self.processed_event_entity_path)


    def entity_entity_raw2df(self,reprocess=True,describe=True):
        """

        找出实体与实体的所有关系，并转化成dataframe格式保存
        原始格式--》嵌套字典格式--》存储格式dataframe

        原始格式
        relation结点 subject entity (头结点)
        relation结点 object entity (尾结点)
        relation结点 roleType 比如参与者(关系)
        relation结点 hasbegintimestamp 时间
        relation结点 hasendtimestamp 时间

        嵌套字典格式
        {relation结点：{头结点，关系，尾结点，开始时间，结束时间}。。。。。。。。。。}

        存储格式
        事件/实体 关系 实体/事件 开始时间 结束时间 如果有空值，则用-1表示，多个关系则随机选取一个
        """
        self._node_relation_datatype_raw2df(reprocess=reprocess,
                                            describe=describe,
                                            datatype="entity_entity",
                                            raw_data_path=self.raw_entity_entity_path,
                                            saved_path=self.processed_entity_entity_path)


    def count_event_node_num(self,reprocess=True,
                             describe=True,
                             event_event=True,
                             event_entity=True):
        """
        统计事件节点的度

        :param reprocess: True重新处理
        :param describe: True展示数据描述
        :param event_event: True统计event-event数据的事件节点数据量
        :param event_entity: True统计event-entity数据的事件节点数据量
        :return: None
        """
        def count_node(rdf_triplets):
            for i,row in tqdm(rdf_triplets.iterrows(),total=len(rdf_triplets)):
                if row["head"] in self.event_dict.keys():
                    self.event_dict[row["head"]]=self.event_dict[row["head"]]+1
                if row["tail"] in self.event_dict.keys():
                    self.event_dict[row["tail"]]=self.event_dict[row["tail"]]+1
        def show(degree):
            print("event_node degree > %d "%(degree),"num",sum(np.array(list(node_rank))>degree),"percent","%.2f%%" % (sum(np.array(list(node_rank))>degree)/event_len * 100))


        if reprocess:
            if event_event:
                count_node(self.rdf_triplets_event_event)
            if event_entity:
                count_node(self.rdf_triplets_event_entity)
            json.dump(self.event_dict, open(self.event_node_count_path, "w"), indent=4, sort_keys=True)
            print("event_node_count has been saved in {}".format(self.event_node_count_path))
        else:
            if os.path.exists(self.event_node_count_path):
                print("loading event_node_count...")
                with open (self.event_node_count_path) as file:
                    self.event_dict = json.load(file)
                print("loading event_node_count succeed!")
            else:
                raise FileNotFoundError("event_node_count_path does not exists!")

        if describe:
            print("top 10 event_node:")
            count_node_rank_item = sorted(self.event_dict.items(), key=lambda x: x[1],reverse=True)
            print(count_node_rank_item[:10])

            node_rank = sorted(self.event_dict.values(),reverse=True)
            event_len=sum(np.array(list(node_rank))>0)
            print("all_event_num",event_len)
            # interval=math.floor(event_len/100)
            # sample_index=np.arange(100)*interval
            # sample_rank=np.array(list(node_rank))[sample_index]
            # print(sample_rank)
            for degree in self.event_degree_list:
                show(degree)

    def count_entity_node_num(self,
                              reprocess=True,
                              describe=True,
                              entity_entity=True,
                              event_entity=True):
        """
        统计实体节点的度

        :param reprocess: True重新处理
        :param describe: True展示数据描述
        :param entity_entity: True统计entity-entity数据的事件节点数据量
        :param event_entity: True统计event-entity数据的事件节点数据量
        :return: None
        """
        def count_node(rdf_triplets):
            for i,row in tqdm(rdf_triplets.iterrows(),total=len(rdf_triplets)):
                if row["head"] in self.entity_dict.keys():
                    self.entity_dict[row["head"]]=self.entity_dict[row["head"]]+1
                if row["tail"] in self.entity_dict.keys():
                    self.entity_dict[row["tail"]]=self.entity_dict[row["tail"]]+1
        def show(degree):
            print("entity_node degree > %d "%(degree),"num",sum(np.array(list(node_rank))>degree),"percent","%.2f%%" % (sum(np.array(list(node_rank))>degree)/entity_len * 100))


        if reprocess:
            if entity_entity:
                count_node(self.rdf_triplets_entity_entity)
            if event_entity:
                count_node(self.rdf_triplets_event_entity)
            json.dump(self.entity_dict, open(self.entity_node_count_path, "w"), indent=4, sort_keys=True)
            print("entity_node_count has been saved in {}".format(self.entity_node_count_path))
        else:
            if os.path.exists(self.entity_node_count_path):
                print("loading entity_node_count...")
                with open (self.entity_node_count_path) as file:
                    self.entity_dict = json.load(file)
                print("loading entity_node_count succeed!")
            else:
                raise FileNotFoundError("entity_node_count_path does not exists!")

        if describe:
            print("top 10 entity_node:")
            count_node_rank_item = sorted(self.entity_dict.items(), key=lambda x: x[1],reverse=True)
            print(count_node_rank_item[:10])

            node_rank = self.entity_dict.values()
            entity_len=sum(np.array(list(node_rank))>0)
            print("all_entity_num",entity_len)
            # interval=math.floor(entity_len/100)
            # sample_index=np.arange(100)*interval
            # sample_rank=np.array(list(node_rank))[sample_index]
            # print(sample_rank)
            for degree in self.entity_degree_list:
                show(degree)

    def filt_event_event(self,event_degree,reprocess=True,describe=True):
        """
        事件-事件 三元组过滤

        :param event_degree: 保留事件节点的度大于degree 的三元组
        """
        if reprocess:
            filt_event_set=set([key for key,value in self.event_dict.items() if value>event_degree])
            print(len(filt_event_set))
            self.filt_triplets_event_event=list()
            for i,row in tqdm(self.rdf_triplets_event_event.iterrows(),total=len(self.rdf_triplets_event_event)):
                if row["head"] in filt_event_set and row["tail"] in filt_event_set:
                    self.filt_triplets_event_event.append(row[1:])
            self.filt_triplets_event_event=pd.DataFrame(self.filt_triplets_event_event)
            self.filt_triplets_event_event.columns=["head","relation","tail","start_time","end_time"]
            self.filt_triplets_event_event.to_csv(self.filt_event_event_path)
            print("filt_triplets_event_event has been saved in {}".format(self.filt_event_event_path))
        else:
            if os.path.exists(self.filt_event_event_path):
                print("loading {}...".format(self.filt_event_event_path))
                self.filt_triplets_event_event = pd.read_csv(self.filt_event_event_path)
                print("loading {} succeed!".format(self.filt_triplets_event_event))
            else:
                raise FileNotFoundError("{} does not exists!".format(self.filt_triplets_event_event))
        if describe:
            print("filt_triplets_event_event_len",len(self.filt_triplets_event_event))
            print("raw_triplets_event_event_len",len(self.rdf_triplets_event_event))
            print("filt_percentage %.2f%%"%(len(self.filt_triplets_event_event)*100/len(self.rdf_triplets_event_event)))

    def filt_event_entity(self,event_degree,entity_degree,reprocess=True,describe=True):
        """
        事件-实体 三元组过滤

        :param event_degree: 保留事件节点的度大于degree 的三元组
        """
        if reprocess:
            filt_entity_set=set([key for key,value in self.entity_dict.items() if value>entity_degree])
            filt_event_set=set([key for key,value in self.event_dict.items() if value>event_degree])
            self.filt_triplets_event_entity=list()
            for i,row in tqdm(self.rdf_triplets_event_entity.iterrows(),total=len(self.rdf_triplets_event_entity)):
                if (row["head"] in filt_event_set and row["tail"] in filt_entity_set) \
                        or (row["head"] in filt_entity_set and row["tail"] in filt_event_set):
                    self.filt_triplets_event_entity.append(row[1:])
            self.filt_triplets_event_entity=pd.DataFrame(self.filt_triplets_event_entity)
            self.filt_triplets_event_entity.columns=["head","relation","tail","start_time","end_time"]
            self.filt_triplets_event_entity.to_csv(self.filt_event_entity_path)
            print("filt_triplets_event_entity has been saved in {}".format(self.filt_event_entity_path))
        else:
            if os.path.exists(self.filt_event_entity_path):
                print("loading {}...".format(self.filt_event_entity_path))
                self.filt_triplets_event_entity = pd.read_csv(self.filt_event_entity_path)
                print("loading {} succeed!".format(self.filt_triplets_event_entity))
            else:
                raise FileNotFoundError("{} does not exists!".format(self.filt_triplets_event_entity))
        if describe:
            print("filt_triplets_event_entity_len",len(self.filt_triplets_event_entity))
            print("raw_triplets_event_entity_len",len(self.rdf_triplets_event_entity))
            print("filt_percentage %.2f%%"%(len(self.filt_triplets_event_entity)*100/len(self.rdf_triplets_event_entity)))

    def filt_entity_entity(self,entity_degree,reprocess=True,describe=True):
        """
        实体-实体 三元组过滤

        :param entity_degree: 保留实体节点的度大于degree 的三元组
        """
        if reprocess:
            filt_entity_set=set([key for key,value in self.entity_dict.items() if value>entity_degree])
            print(len(filt_entity_set))
            self.filt_triplets_entity_entity=list()
            for i,row in tqdm(self.rdf_triplets_entity_entity.iterrows(),total=len(self.rdf_triplets_entity_entity)):
                if row["head"] in filt_entity_set and row["tail"] in filt_entity_set:
                    self.filt_triplets_entity_entity.append(row[1:])
            self.filt_triplets_entity_entity=pd.DataFrame(self.filt_triplets_entity_entity)
            self.filt_triplets_entity_entity.columns=["head","relation","tail","start_time","end_time"]
            self.filt_triplets_entity_entity.to_csv(self.filt_entity_entity_path)
            print("filt_triplets_entity_entity has been saved in {}".format(self.filt_entity_entity_path))
        else:
            if os.path.exists(self.filt_entity_entity_path):
                print("loading {}...".format(self.filt_entity_entity_path))
                self.filt_triplets_entity_entity = pd.read_csv(self.filt_entity_entity_path)
                print("loading {} succeed!".format(self.filt_triplets_entity_entity))
            else:
                raise FileNotFoundError("{} does not exists!".format(self.filt_triplets_entity_entity))
        if describe:
            print("filt_triplets_entity_entity_len",len(self.filt_triplets_entity_entity))
            print("raw_triplets_entity_entity_len",len(self.rdf_triplets_entity_entity))
            print("filt_percentage %.2f%%"%(len(self.filt_triplets_entity_entity)*100/len(self.rdf_triplets_entity_entity)))

    def create_subeventkg_rdf2name_all(self,event_degree,entity_degree,reprocess=True):
        """
        建立rdf转name的三个字典
        """
        if reprocess:
            filt_event_set=set([key for key,value in self.event_dict.items() if value>event_degree])
            filt_entity_set=set([key for key,value in self.entity_dict.items() if value>entity_degree])
            filt_relation_set=set()
            for i,row in tqdm(self.rdf_triplets_event_event.iterrows(),total=len(self.rdf_triplets_event_event)):
                if row["head"] in filt_event_set and row["tail"] in filt_event_set:
                    filt_relation_set.add(row[2])
            for i,row in tqdm(self.rdf_triplets_event_entity.iterrows(),total=len(self.rdf_triplets_event_entity)):
                if (row["head"] in filt_event_set and row["tail"] in filt_entity_set) \
                        or (row["head"] in filt_entity_set and row["tail"] in filt_event_set):
                    filt_relation_set.add(row[2])
            for i,row in tqdm(self.rdf_triplets_entity_entity.iterrows(),total=len(self.rdf_triplets_entity_entity)):
                if row["head"] in filt_entity_set and row["tail"] in filt_entity_set:
                    filt_relation_set.add(row[2])

            index=0
            for event in tqdm(filt_event_set):
                self.event_rdf2name_dict[event]="Q_{}".format(index)
                index=index+1
            index=0
            for entity in tqdm(filt_entity_set):
                self.entity_rdf2name_dict[entity]="E_{}".format(index)
                index=index+1
            index=0
            for relation in tqdm(filt_relation_set):
                self.relation_rdf2name_dict[relation]="R_{}".format(index)
                index=index+1

            json.dump(self.event_rdf2name_dict, open(self.event_rdf2name_path, "w"), indent=4, sort_keys=True)
            print("event_rdf2name has been saved in {}".format(self.event_rdf2name_path))
            json.dump(self.entity_rdf2name_dict, open(self.entity_rdf2name_path, "w"), indent=4, sort_keys=True)
            print("entity_rdf2name has been saved in {}".format(self.entity_rdf2name_path))
            json.dump(self.relation_rdf2name_dict, open(self.relation_rdf2name_path, "w"), indent=4, sort_keys=True)
            print("relation_rdf2name has been saved in {}".format(self.relation_rdf2name_path))
            print("event_rdf2name_dict_len",len(self.event_rdf2name_dict))
            print("entity_rdf2name_dict_len",len(self.entity_rdf2name_dict))
            print("relation_rdf2name_dict_len",len(self.relation_rdf2name_dict))

        else:
            print("loading {}...".format(self.event_rdf2name_path))
            with open(self.event_rdf2name_path) as file:
                self.event_rdf2name_dict=json.load(file)
            print("event_rdf2name_dict_len",len(self.event_rdf2name_dict))

            print("loading {}...".format(self.entity_rdf2name_path))
            with open(self.entity_rdf2name_path) as file:
                self.entity_rdf2name_dict=json.load(file)
            print("entity_rdf2name_dict_len",len(self.entity_rdf2name_dict))

            print("loading {}...".format(self.relation_rdf2name_path))
            with open(self.relation_rdf2name_path) as file:
                self.relation_rdf2name_dict=json.load(file)
            print("relation_rdf2name_dict_len",len(self.relation_rdf2name_dict))

    def create_relation_rdf2type_name(self,reprocess):
        if reprocess:
            self.type_name_dict=dict()
            file_list=["property_labels.nq","type_labels_dbpedia.nq"]
            for single_file in file_list:
                with open(single_file,"r",encoding="utf-8") as file:
                    print("creating relation_rdf2name...")
                    for line in tqdm(file, total=self._get_num_lines(single_file)):
                        x=line.strip().split(" ")
                        if x[1]=="<http://www.w3.org/2000/01/rdf-schema#label>":
                            match_instance = re.findall(r"^<.*?> <.*?> (\".*?\"@en) <.*?> \D", line)
                            if len(match_instance) != 0:
                                self.type_name_dict[x[0]]= match_instance[0][1:-4]
            json.dump(self.type_name_dict,open("rdf2type_name","w"),indent=4,sort_keys=True)

        else:
            with open("rdf2type_name") as file:
                self.type_name_dict=json.load(file)


    def create_subeventkg_event_lut(self,event_degree,reprocess=True):
        if reprocess:
            filt_event_set=set([key for key,value in self.event_dict.items() if value>event_degree])
            event_lut=defaultdict(dict)
            for key,value in self.event_rdf2name_dict.items():
                event_lut[value]["name"] = -1
                event_lut[value]["name_rdf"] =  key
                event_lut[value]["type"] = -1
                event_lut[value]["type_rdf"] = -1
                event_lut[value]["description"] = -1
                # event_lut[value]["begintime"] = -1
                # event_lut[value]["endtime"] = -1

            with open(self.raw_events_path,"r",encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines(self.raw_events_path)):
                    x=line.strip().split(" ")
                    if x[1]=="<http://www.w3.org/2000/01/rdf-schema#label>" and x[0] in filt_event_set:
                        match_instance = re.findall(r"^<.*?> <.*?> (\".*?\"@en) <.*?> \D", line)
                        if len(match_instance) != 0:
                            event_lut[self.event_rdf2name_dict[x[0]]]["name"] = match_instance[0][1:-4]
                    if x[1]=="<http://purl.org/dc/terms/description>" and x[0] in filt_event_set:
                        match_instance = re.findall(r"^<.*?> <.*?> (\".*?\"@en) <.*?> \D", line)
                        if len(match_instance) != 0:
                            event_lut[self.event_rdf2name_dict[x[0]]]["description"] = match_instance[0][1:-4]


            with open("types_dbpedia.nq","r",encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines("types_dbpedia.nq")):
                    x=line.strip().split(" ")
                    if x[1]=="<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>" and x[0] in filt_event_set:
                        event_lut[self.event_rdf2name_dict[x[0]]]["type_rdf"] = x[2]
                        event_lut[self.event_rdf2name_dict[x[0]]]["type"] = self.type_dict[x[2]]
            with open("relations_base.nq","r",encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines("relations_base.nq")):
                    x=line.strip().split(" ")
                    # if x[1]=="<http://semanticweb.cs.vu.nl/2009/11/sem/hasBeginTimeStamp>" and x[0] in filt_event_set:
                    #     event_lut[self.event_rdf2name_dict[x[0]]]["begintime"] = x[2][1:11]
                    # if x[1] == "<http://semanticweb.cs.vu.nl/2009/11/sem/hasEndTimeStamp>" and x[0] in filt_event_set:
                    #     event_lut[self.event_rdf2name_dict[x[0]]]["endtime"] = x[2][1:11]

            json.dump(event_lut,open(self.event_lut_path,"w"),indent=4,sort_keys=True)

    def create_subeventkg_entity_lut(self,entity_degree,reprocess=True):
        if reprocess:
            filt_entity_set=set([key for key,value in self.entity_dict.items() if value>entity_degree])
            entity_lut=defaultdict(dict)
            for key,value in self.entity_rdf2name_dict.items():
                entity_lut[value]["name"] = -1
                entity_lut[value]["name_rdf"] =  key
                entity_lut[value]["type"] = -1
                entity_lut[value]["type_rdf"] = -1
                entity_lut[value]["description"] = -1
                # entity_lut[value]["begintime"] = -1
                # entity_lut[value]["endtime"] = -1

            with open(self.raw_entities_path,"r",encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines(self.raw_entities_path)):
                    x=line.strip().split(" ")
                    if x[1]=="<http://www.w3.org/2000/01/rdf-schema#label>" and x[0] in filt_entity_set:
                        match_instance = re.findall(r"^<.*?> <.*?> (\".*?\"@en) <.*?> \D", line)
                        if len(match_instance) != 0:
                            entity_lut[self.entity_rdf2name_dict[x[0]]]["name"] = match_instance[0][1:-4]


            file_list=["types_dbpedia.nq","types.nq"]
            for single_file in file_list:
                with open(single_file,"r",encoding="utf-8") as file:
                    for line in tqdm(file, total=self._get_num_lines(single_file)):
                        x=line.strip().split(" ")
                        if x[1]=="<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>" and x[0] in filt_entity_set:
                            entity_lut[self.entity_rdf2name_dict[x[0]]]["type_rdf"] = x[2]
                            entity_lut[self.entity_rdf2name_dict[x[0]]]["type"] =self.type_dict[x[2]]
            with open("relations_base.nq","r",encoding="utf-8") as file:
                for line in tqdm(file, total=self._get_num_lines("relations_base.nq")):
                    x=line.strip().split(" ")
                    # if x[1]=="<http://semanticweb.cs.vu.nl/2009/11/sem/hasBeginTimeStamp>" and x[0] in filt_entity_set:
                    #     entity_lut[self.entity_rdf2name_dict[x[0]]]["begintime"] = x[2][1:11]
                    # if x[1] == "<http://semanticweb.cs.vu.nl/2009/11/sem/hasEndTimeStamp>" and x[0] in filt_entity_set:
                    #     entity_lut[self.entity_rdf2name_dict[x[0]]]["endtime"] = x[2][1:11]

            print(entity_lut)
            json.dump(entity_lut,open(self.entity_lut_path,"w"),indent=4,sort_keys=True)

    def create_subeventkg_relation_lut(self,reprocess=True):
        if reprocess:
            print("开始")
            my_dict=defaultdict(dict)
            for key,value in self.relation_rdf2name_dict.items():
                my_dict[value]["rdf"] = key
                my_dict[value]["name"] = -1
            for key,value in self.relation_rdf2name_dict.items():
                my_dict[value]["rdf"] = key
                try:
                    my_dict[value]["name"] = self.type_name_dict[key]
                except Exception as e:
                    match_instance = re.findall(r"<.*?/.*?/.*?/.*?/(.*?)>$", key)
                    my_dict[value]["name"] =  match_instance[0]
            json.dump(my_dict,open(self.relation_lut_path,"w"),indent=4,sort_keys=True)

    def merge_all_data_convert_name(self,reprocess=True,event_degree=10,entity_degree=10):
        filt_event_set=set([key for key,value in self.event_dict.items() if value>event_degree])
        filt_entity_set=set([key for key,value in self.entity_dict.items() if value>entity_degree])
        if reprocess:
            df_lines=[]
            self.total_rdf_triplets=pd.concat([self.filt_triplets_event_event,
                                               self.filt_triplets_event_entity,
                                               self.filt_triplets_entity_entity],axis=0)
            for i,row in tqdm(self.total_rdf_triplets.iterrows(),total=len(self.total_rdf_triplets)):
                if row["head"] in filt_event_set:
                    new_head=self.event_rdf2name_dict[row["head"]]
                if row["head"] in filt_entity_set:
                    new_head=self.entity_rdf2name_dict[row["head"]]
                new_relation=self.relation_rdf2name_dict[row["relation"]]
                if row["tail"] in filt_event_set:
                    new_tail=self.event_rdf2name_dict[row["tail"]]
                if row["tail"] in filt_entity_set:
                    new_tail=self.entity_rdf2name_dict[row["tail"]]
                if str(row["start_time"])!="-1":
                    new_start_time=row["start_time"][1:11]
                else:
                    new_start_time=row["start_time"]
                if str(row["end_time"])!="-1":
                    new_end_time=row["end_time"][1:11]
                else:
                    new_end_time=row["end_time"]

                df_lines.append([new_head,new_relation,new_tail,new_start_time,new_end_time])

            self.triplets=pd.DataFrame(df_lines)
            self.triplets.columns=["head","relation","tail","start_time","end_time"]
            self.triplets.to_csv("all_triplets_data.txt",index=None)
        else:
            self.triplets=pd.read_csv("all_triplets_data.txt",low_memory=False)

    def split_data(self,reprocess=True,seed=1,threshold=5,test_num=50000):
        if reprocess:
            node_list=list()
            self.triplets = shuffle(self.triplets)
            # self.triplets=self.triplets.sample(frac=1).reset_index(drop=True)
            for i,row in tqdm(self.triplets.iterrows(),total=len(self.triplets)):
                node_list.append(row["head"])
                node_list.append(row["tail"])
            node_count_dict=Counter(node_list)
            train_list=[]
            valid_list=[]
            test_list=[]

            random.seed(seed)
            for i,row in tqdm(self.triplets.iterrows(),total=len(self.triplets)):
                if node_count_dict[row["head"]]>threshold and node_count_dict[row["tail"]]>threshold and len(test_list)<test_num:
                    if random.random()<0.5:
                        valid_list.append(row)
                    else:
                        test_list.append(row)
                    node_count_dict[row["head"]]=node_count_dict[row["head"]]-1
                    node_count_dict[row["tail"]]=node_count_dict[row["tail"]]-1
                else:
                    train_list.append(row)
            print("len(train_list)",len(train_list))
            print("len(valid_list)",len(valid_list))
            print("len(test_list)",len(test_list))

            df_train = pd.DataFrame(train_list)
            # df_train.columns=["head","relation","tail","start_time","end_time"]
            df_train.to_csv("eventkg2m_train.txt",index=None,header=None,sep="\t")

            df_valid = pd.DataFrame(valid_list)
            # df_valid.columns=["head","relation","tail","start_time","end_time"]
            df_valid.to_csv("eventkg2m_valid.txt",index=None,header=None,sep="\t")

            df_test = pd.DataFrame(test_list)
            # df_test.columns=["head","relation","tail","start_time","end_time"]
            df_test.to_csv("eventkg2m_test.txt",index=None,header=None,sep="\t")


subeventkg_processor=SUBEVENTKG_Processor(events_nq_path="events.nq",
                                          entities_nq_path="entities.nq",
                                          relations_base_nq_path="relations_base.nq",
                                          relations_events_other_nq_path="relations_events_other.nq",
                                          relations_entities_temporal_nq_path="relations_entities_temporal.nq",
                                          processed_entities_path="processed_entity.json",
                                          processed_events_path="processed_event.json",
                                          processed_event_event_path="processed_event_event.csv",
                                          processed_event_entity_path="processed_event_entity.csv",
                                          processed_entity_entity_path="processed_entity_entity.csv",
                                          filt_event_event_path="filt_event_event.csv",
                                          filt_event_entity_path="filt_event_entity.csv",
                                          filt_entity_entity_path="filt_entity_entity.csv",
                                          event_node_count_path="event_node_count_path.json",
                                          entity_node_count_path="entity_node_count_path.json",
                                          event_rdf2name_path="event_rdf2name_path.json",
                                          entity_rdf2name_path="entity_rdf2name_path.json",
                                          relation_rdf2name_path="relation_rdf2name_path.json",
                                          event_lut_path="eventkg2m_events_lut.json",
                                          entity_lut_path="eventkg2m_entities_lut.json",
                                          relation_lut_path="eventkg2m_relations_lut.json",
                                          event_degree_list=[0,3,5,10,30,50,100,200,500,1000,2000],
                                          entity_degree_list=[0,3,5,10,30,50,70,100],
                                          )
subeventkg_processor.create_entities_index(reprocess=False,describe=True)
subeventkg_processor.create_events_index(reprocess=False,describe=True)
subeventkg_processor.event_event_raw2df(reprocess=False,describe=True)
subeventkg_processor.event_entity_raw2df(reprocess=False,describe=True)
subeventkg_processor.entity_entity_raw2df(reprocess=False,describe=True)
subeventkg_processor.count_event_node_num(reprocess=False,describe=False,
                                          event_event=True,event_entity=True)
subeventkg_processor.count_entity_node_num(reprocess=False,describe=False,
                                           entity_entity=True,event_entity=True)
subeventkg_processor.filt_event_event(reprocess=False,describe=True,
                                      event_degree=10)
subeventkg_processor.filt_event_entity(reprocess=False,describe=True,
                                       event_degree=10,entity_degree=10)
subeventkg_processor.filt_entity_entity(reprocess=False,describe=True,
                                        entity_degree=10)
subeventkg_processor.create_subeventkg_rdf2name_all(reprocess=True,event_degree=10,entity_degree=10)
subeventkg_processor.create_relation_rdf2type_name(reprocess=True)
subeventkg_processor.create_subeventkg_event_lut(reprocess=False,event_degree=10)
subeventkg_processor.create_subeventkg_entity_lut(reprocess=False,entity_degree=10)
subeventkg_processor.create_subeventkg_relation_lut(reprocess=True)
subeventkg_processor.merge_all_data_convert_name(reprocess=True,event_degree=10,entity_degree=10)
subeventkg_processor.split_data(reprocess=True,seed=1,threshold=8,test_num=20000)

"""
all_event_num 679 953
event_node degree > 0  num 679953 percent 100.00%
event_node degree > 3  num 343086 percent 50.46%
event_node degree > 5  num 265090 percent 38.99%
event_node degree > 10  num 115579 percent 17.00%(对勾)
event_node degree > 30  num 15030 percent 2.21%
event_node degree > 50  num 7038 percent 1.04%
event_node degree > 100  num 2584 percent 0.38%
event_node degree > 200  num 1177 percent 0.17%
event_node degree > 500  num 473 percent 0.07%
event_node degree > 1000  num 227 percent 0.03%
event_node degree > 2000  num 133 percent 0.02%
"""

"""
all_entity_num 3 175 188
entity_node degree > 0  num 3175188 percent 100.00%
entity_node degree > 3  num 504056 percent 15.87%
entity_node degree > 5  num 288365 percent 9.08%
entity_node degree > 10  num 135494 percent 4.27%(对勾)
entity_node degree > 30  num 28599 percent 0.90%
entity_node degree > 50  num 16965 percent 0.53%
entity_node degree > 70  num 11965 percent 0.38%
entity_node degree > 100  num 8205 percent 0.26%
entity_node degree > 500  num 1497 percent 0.05%
entity_node degree > 1000  num 606 percent 0.02%
entity_node degree > 2000  num 201 percent 0.01%
entity_node degree > 5000  num 72 percent 0.00%
entity_node degree > 10000  num 30 percent 0.00%
entity_node degree > 20000  num 16 percent 0.00%
entity_node degree > 10000  num 30 percent 0.00%
"""
"""
event_degree=10,entity_degree=10(对勾)
event  115 579
entity 135 494
relation   822
822
event-event       221 944 /  730 830   30.37%
event-entity    1 140 552 /4 270 521   26.71%
entity-entity     971 490 /4 361 809   22.27%
total_triplets 2 333 986
total_node       251 073
三元组比节点的比例 9.23:1
训练集 2 200 000左右,实际2 294 008
验证集    20 000左右,实际   19 978
测试集    20 000左右,实际   20 000
"""






