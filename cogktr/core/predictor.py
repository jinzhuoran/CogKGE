import copy
import datetime
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from mongoengine import StringField, IntField, FloatField, BooleanField, DateTimeField, Document
from mongoengine import connect
from mongoengine.queryset.visitor import Q
from openTSNE import TSNE
from tqdm import tqdm


class Kr_Predictior:
    def __init__(self,
                 model,
                 pretrained_model_path,
                 model_name,
                 data_name,
                 device,
                 node_lut,
                 relation_lut,
                 processed_data_path="data",
                 reprocess=True,
                 fuzzy_query_top_k=10,
                 predict_top_k=50):
        """
        三元组链接预测
        :param model: 模型
        :param pretrained_model_path: 知识表示预训练模型路径
        :param model_name: 模型名字
        :param data_name: 数据集名字
        :param device: 显卡编号
        :param node_lut: 节点查找表
        :param relation_lut: 关系查找表
        :param processed_data_path: 处理后的数据输出路径
        :param reprocess: 是否重新处理
        :param fuzzy_query_top_k: 模糊查询前fuzzy_query_top_k个最相似的结果
        :param predict_top_k: 链接预测前predict_top_k的结果
        """

        if not isinstance(reprocess, bool):
            raise TypeError("param reprocess is True or False!")
        if not isinstance(fuzzy_query_top_k, int):
            raise TypeError("param fuzzy_query_top_k is not int type!")
        if not (1 <= fuzzy_query_top_k):
            raise ValueError("param fuzzy_query_top_k must be greater than 0!")
        if not isinstance(predict_top_k, int):
            raise TypeError("param predict_top_k is not int type!")
        if not (1 <= predict_top_k):
            raise ValueError("param predict_top_k must be greater than 0!")
        if not os.path.exists(pretrained_model_path):
            raise FileExistsError("pretrained_model_path doesn't exist!")
        if not os.path.exists(processed_data_path):
            raise FileExistsError("processed_data_path doesn't exist!")
        if not os.path.exists(os.path.join(processed_data_path, data_name, model_name)):
            os.makedirs(os.path.join(processed_data_path, data_name, model_name))

        self.model = model
        self.pretrained_mode_path = pretrained_model_path
        self.model_name = model_name
        self.data_name = data_name
        self.device = device
        self.node_lut = node_lut
        self.relation_lut = relation_lut
        self.processed_data_path = os.path.join(processed_data_path, self.data_name, model_name)
        self.reprocess = reprocess
        self.fuzzy_query_top_k = fuzzy_query_top_k
        self.predict_top_k = predict_top_k

        self.node_len = len(self.node_lut)
        self.relation_len = len(self.relation_lut)
        self.summary_node_dict = defaultdict(dict)  # 模糊查询节点字典
        self.summary_relation_dict = defaultdict(dict)  # 模糊查询关系字典
        self.detailed_node_dict = defaultdict(dict)  # 链接预测节点字典
        self.detailed_relation_dict = defaultdict(dict)  # 链接预测关系字典
        self.summary_node_dict_path = os.path.join(self.processed_data_path, "summary_node_dict.json")
        self.summary_relation_dict_path = os.path.join(self.processed_data_path, "summary_relation_dict.json")
        self.detailed_node_dict_path = os.path.join(self.processed_data_path, "detailed_node_dict.json")
        self.detailed_relation_dict_path = os.path.join(self.processed_data_path, "detailed_relation_dict.json")
        self.model.load_state_dict(torch.load(self.pretrained_mode_path))
        self.model = self.model.to(self.device)
        self.all_node_index_column_matrix = torch.unsqueeze(torch.arange(self.node_len).to(self.device), dim=1)
        self.all_node_index_row_vector = torch.arange(self.node_len).to(self.device)
        self.all_node_embedding = self.model.entity_embedding_base(self.all_node_index_row_vector)
        self.all_relation_index_column_matrix = torch.unsqueeze(torch.arange(self.relation_len).to(self.device), dim=1)

        self._create_summary_dict()  # 建立模糊查询字典
        self._create_detailed_dict()  # 建立链接预测字典

    def _create_summary_dict(self):
        """
        建立模糊查询字典
        """
        if self.reprocess:
            print("Creating_summary_node_dict...")
            for i in tqdm(range(self.node_len)):
                item_df = self.node_lut[i]
                item = {}
                item["id"] = int(str(item_df["name_id"]))
                item["name"] = item_df["name"]
                item["summary"] = item_df["description"]
                self.summary_node_dict[str(item_df["name_id"])] = item
            json.dump(self.summary_node_dict, open(self.summary_node_dict_path, "w"), indent=4, sort_keys=False)
            print("Creating_summary_relation_dict...")
            for i in tqdm(range(self.relation_len)):
                item_df = self.relation_lut[i]
                item = {}
                item["id"] = int(str(item_df["name_id"]))
                item["name"] = item_df["name"]
                item["summary"] = item_df["rdf"]
                self.summary_relation_dict[str(item_df["name_id"])] = item
            json.dump(self.summary_relation_dict, open(self.summary_relation_dict_path, "w"), indent=4, sort_keys=False)
        else:
            if os.path.exists(self.summary_node_dict_path):
                with open(self.summary_node_dict_path) as file:
                    self.summary_node_dict = json.load(file)
            else:
                raise FileExistsError("{} does not exist!".format(self.summary_node_dict_path))
            if os.path.exists(self.summary_relation_dict_path):
                with open(self.summary_relation_dict_path) as file:
                    self.summary_relation_dict = json.load(file)
            else:
                raise FileExistsError("{} does not exist!".format(self.summary_relation_dict_path))

    def _create_detailed_dict(self):
        """
        建立链接预测字典
        """
        if self.reprocess:
            print("Creating_detailed_node_dict...")
            for i in tqdm(range(self.node_len)):
                item_df = self.node_lut[i]
                item = {}
                item["id"] = int(str(item_df["name_id"]))
                item["name"] = item_df["name"]
                item["rdf"] = item_df["name_rdf"]
                item["type"] = item_df["type"]
                item["type_rdf"] = item_df["type_rdf"]
                item["node_type"] = item_df["node_type"]
                item["description"] = item_df["description"]
                self.detailed_node_dict[str(item_df["name_id"])] = item
            json.dump(self.detailed_node_dict, open(self.detailed_node_dict_path, "w"), indent=4, sort_keys=False)
            print("Creating_detailed_relation_dict...")
            for i in tqdm(range(self.relation_len)):
                item_df = self.relation_lut[i]
                item = {}
                item["id"] = int(str(item_df["name_id"]))
                item["name"] = item_df["name"]
                item["summary"] = item_df["rdf"]
                self.detailed_relation_dict[str(item_df["name_id"])] = item
            json.dump(self.detailed_relation_dict, open(self.detailed_relation_dict_path, "w"), indent=4,
                      sort_keys=False)
        else:
            if os.path.exists(self.detailed_node_dict_path):
                with open(self.detailed_node_dict_path) as file:
                    self.detailed_node_dict = json.load(file)
            else:
                raise FileExistsError("{} does not exist!".format(self.detailed_node_dict_path))
            if os.path.exists(self.detailed_relation_dict_path):
                with open(self.detailed_relation_dict_path) as file:
                    self.detailed_relation_dict = json.load(file)
            else:
                raise FileExistsError("{} does not exist!".format(self.detailed_relation_dict_path))

    def insert_entity(self, entity):
        # {'name': 'Tom Waits', 'rdf': '<http://eventKG.l3s.uni-hannover.de/resource/entity_3481409>', 'type': 'Singer',
        #  'type_rdf': '<http://dbpedia.org/ontology/Singer>', 'node_type': 'entity', 'description': '-1', 'idd': '1'}
        Entity.objects.create(**entity)

    def insert_relation(self, relation):
        # relation = {'name': 'hasWonPrize', 'summary': '<http://yago-knowledge.org/resource/hasWonPrize>', 'idd': '1'}
        Relation.objects.create(**relation)

    def remove_all(self):
        Entity.objects(name__contains="").delete()
        Relation.objects(name__contains="").delete()

    def fuzzy_query_node_keyword(self, node_keyword=None):
        """
        模糊查询节点名字
        :param node_keyword: 输入节点名字,如果输入为空，则返回10个范例
        :return: 模糊节点列表
        """
        if node_keyword is None:
            node_keyword = ''
        # entities = Entity.objects(Q(name__contains=node_keyword) | Q(
        #     description__contains=node_keyword))
        entities = Entity.objects(name__contains=str(node_keyword)).limit(self.fuzzy_query_top_k)
        results = []
        for entity in entities:
            results.append(entity.to_dict())
        return results

    def fuzzy_query_relation_keyword(self, relation_keyword=None):
        """
        模糊查询关系名字
        :param relation_keyword: 输入关系名字，如果输入为空，则返回10个范例
        :return: 模糊关系列表
        """
        if relation_keyword is None:
            relation_keyword = ''
        relations = Relation.objects(Q(name__contains=str(relation_keyword)) | Q(
            summary__contains=str(relation_keyword))).limit(self.fuzzy_query_top_k)
        results = []
        for relation in relations:
            results.append(relation.to_dict())
        return results

    def predict_similar_node(self, node_id):
        """
        预测相似的节点
        :param node_id: 输入节点id
        :return: similar_node_list 相似节点列表
        """

        node_embedding = torch.unsqueeze(self.model.entity_embedding_base(torch.tensor(node_id).to(self.device)), dim=0)
        distance = F.pairwise_distance(node_embedding, self.all_node_embedding, p=2)
        value, index = torch.topk(distance, self.predict_top_k + 1, largest=False)
        value = value[1:]
        index = index[1:]
        similar_node_list = []
        value = to_confidence(value)
        for i in range(self.predict_top_k):
            id = str(int(index[i]))
            item = copy.deepcopy(self.detailed_node_dict[id])
            item["confidence"] = value[i].item()
            similar_node_list.append(item)
        return similar_node_list

    def predcit_tail(self, head_id, relation_id=None):
        """
        根据头节点和关系预测尾节点
        如果关系为空，则遍历所有关系，计算出每种关系得分最高的，选出前topk个节点
        :param head_id: 输入头节点id
        :param relation_id: 输入关系id
        :return: tail_list 尾实体列表
        """
        if relation_id is None:
            with torch.no_grad():
                score_list = []  # 每一个位置i对应关系i最好的得分
                tail_index_list = []  # 每一个位置i对应关系i最好的尾节点id
                for relation_id in range(self.relation_len):
                    head_id_vector = torch.tensor(head_id).expand(self.node_len, 1).to(self.device)
                    relation_id_vector = torch.tensor(relation_id).expand(self.node_len, 1).to(self.device)
                    tail_id_vectoe = self.all_node_index_column_matrix.clone()
                    triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vectoe], dim=1)
                    distance = self.model(triplet_id_matric)
                    tail_index = distance.argmin(dim=0)  # 关系i score最好的 尾节点id
                    score = distance[tail_index]  # 关系i score最好的 得分数值
                    score_list.append(score)
                    tail_index_list.append(tail_index)
                score = torch.tensor(score_list)
                tail_index = torch.tensor(tail_index_list)
                value, relation_index = torch.topk(score, self.predict_top_k, largest=False)
                tail_list = []
                value = to_confidence(value)
                for i in range(self.predict_top_k):
                    tail_id = int(tail_index[i])
                    relation_id = int(relation_index[i])
                    item = copy.deepcopy(self.detailed_node_dict[str(tail_id)])
                    item["confidence"] = value[i].item()
                    item["triplet_id"] = (head_id, relation_id, tail_id)
                    item["triplet_name"] = (self.node_lut[head_id]["name"], self.relation_lut[relation_id]["name"],
                                            self.node_lut[tail_id]["name"])
                    tail_list.append(item)
                return tail_list
        else:
            head_id_vector = torch.tensor(head_id).expand(self.node_len, 1).to(self.device)
            relation_id_vector = torch.tensor(relation_id).expand(self.node_len, 1).to(self.device)
            tail_id_vectoe = self.all_node_index_column_matrix.clone()
            triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vectoe], dim=1)
            distance = self.model(triplet_id_matric)
            value, index = torch.topk(distance, self.predict_top_k, largest=False)
            tail_list = []
            value = to_confidence(value)
            for i in range(self.predict_top_k):
                tail_id = int(index[i])
                item = copy.deepcopy(self.detailed_node_dict[str(tail_id)])
                item["confidence"] = value[i].item()
                item["triplet_id"] = (head_id, relation_id, tail_id)
                item["triplet_name"] = (
                    self.node_lut[head_id]["name"], self.relation_lut[relation_id]["name"],
                    self.node_lut[tail_id]["name"])
                tail_list.append(item)
            return tail_list

    def predict_head(self, tail_id, relation_id=None):
        """
        根据尾节点和关系预测头节点
        如果关系为空，则遍历所有关系，计算出每种关系得分最高的，选出前topk个节点
        :param tail_id: 输入尾节点id
        :param relation_id: 输入关系id
        :return: head_list 头节点列表
        """
        if relation_id is None:
            with torch.no_grad():
                score_list = []  # 每一个位置i对应关系i最好的得分
                head_index_list = []  # 每一个位置i对应关系i最好的头节点id
                for relation_id in range(self.relation_len):
                    tail_id_vector = torch.tensor(tail_id).expand(self.node_len, 1).to(self.device)
                    relation_id_vector = torch.tensor(relation_id).expand(self.node_len, 1).to(self.device)
                    head_id_vector = self.all_node_index_column_matrix.clone()
                    triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vector], dim=1)
                    distance = self.model(triplet_id_matric)
                    head_index = distance.argmin(dim=0)  # 关系i score最好的 头节点id
                    score = distance[head_index]  # 关系i score最好的 得分数值
                    score_list.append(score)
                    head_index_list.append(head_index)
                score = torch.tensor(score_list)
                head_index = torch.tensor(head_index_list)
                value, relation_index = torch.topk(score, self.predict_top_k, largest=False)
                head_list = []
                value = to_confidence(value)
                for i in range(self.predict_top_k):
                    head_id = int(head_index[i])
                    relation_id = int(relation_index[i])
                    item = copy.deepcopy(self.detailed_node_dict[str(head_id)])
                    item["confidence"] = value[i].item()
                    item["triplet_id"] = (head_id, relation_id, tail_id)
                    item["triplet_name"] = (self.node_lut[head_id]["name"], self.relation_lut[relation_id]["name"],
                                            self.node_lut[tail_id]["name"])
                    head_list.append(item)
                return head_list
        else:
            tail_id_vector = torch.tensor(tail_id).expand(self.node_len, 1).to(self.device)
            relation_id_vector = torch.tensor(relation_id).expand(self.node_len, 1).to(self.device)
            head_id_vector = self.all_node_index_column_matrix.clone()
            triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vector], dim=1)
            distance = self.model(triplet_id_matric)
            value, index = torch.topk(distance, self.predict_top_k, largest=False)
            node_list = []
            value = to_confidence(value)
            for i in range(self.predict_top_k):
                head_id = int(index[i])
                item = copy.deepcopy(self.detailed_node_dict[str(head_id)])
                item["confidence"] = value[i].item()
                item["triplet_id"] = (head_id, relation_id, tail_id)
                item["triplet_name"] = (
                    self.node_lut[head_id]["name"], self.relation_lut[relation_id]["name"],
                    self.node_lut[tail_id]["name"])
                node_list.append(item)
            return node_list

    def predict_relation(self, head_id, tail_id):
        """
        根据头节点和尾节点预测关系
        :param head_id: 输入头节点id
        :param tail_id: 输入尾节点id
        :return: relation_list 关系列表
        """

        head_id_vector = torch.tensor(head_id).expand(self.relation_len, 1).to(self.device)
        relation_id_vector = self.all_relation_index_column_matrix.clone()
        tail_id_vector = torch.tensor(tail_id).expand(self.relation_len, 1).to(self.device)
        triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vector], dim=1)
        distance = self.model(triplet_id_matric)
        value, index = torch.topk(distance, self.predict_top_k, largest=False)
        relation_list = []
        value = to_confidence(value)
        for i in range(self.predict_top_k):
            relation_id = int(index[i])
            item = copy.deepcopy(self.detailed_relation_dict[str(relation_id)])
            item["confidence"] = value[i].item()
            item["triplet_id"] = (head_id, relation_id, tail_id)
            item["triplet_name"] = (
                self.node_lut[head_id]["name"], self.relation_lut[relation_id]["name"], self.node_lut[tail_id]["name"])
            relation_list.append(item)

        return relation_list

    def show_img(self, node_id, visual_num=1000, filt=True):
        node_embedding = torch.unsqueeze(self.model.entity_embedding_base(torch.tensor(node_id).to(self.device)), dim=0)
        distance = F.pairwise_distance(node_embedding, self.all_node_embedding, p=2)
        value, index = torch.topk(distance, visual_num, largest=False)
        embedding = self.model.entity_embedding_base.weight.data[index].clone().cpu().numpy()
        embedding = TSNE(negative_gradient_method="bh").fit(embedding)
        label_dict = defaultdict(dict)
        label_set = set()
        for i in range(visual_num):
            id = str(int(index[i]))
            item = copy.deepcopy(self.detailed_node_dict[id])
            if item["type"] not in label_set:
                label_dict[item["type"]]["label"] = item["type"]
                label_set.add(item["type"])
                label_dict[item["type"]]["data"] = list()
            label_dict[item["type"]]["data"].append(
                {"x": embedding[i][0], "y": embedding[i][1], "name": item["name"], "confidence": value[i].item()})
        visual_list = list()
        for key, value in label_dict.items():
            visual_list.append(value)

        return visual_list


class Entity(Document):
    idd = StringField(required=True)
    name = StringField(required=True)
    rdf = StringField(required=True)
    type = StringField(required=True)
    type_rdf = StringField(required=True)
    node_type = StringField(required=True)
    description = StringField(required=True)

    def to_dict(self):
        return to_dict_helper(self)


class Relation(Document):
    idd = StringField(required=True)
    name = StringField(required=True)
    summary = StringField(required=True)

    def to_dict(self):
        return to_dict_helper(self)


def to_dict_helper(obj):
    return_data = []
    for field_name in obj._fields:
        if field_name in ("id",):
            continue
        data = obj._data[field_name]
        if isinstance(obj._fields[field_name], StringField):
            return_data.append((field_name, str(data)))
        elif isinstance(obj._fields[field_name], FloatField):
            return_data.append((field_name, float(data)))
        elif isinstance(obj._fields[field_name], IntField):
            return_data.append((field_name, int(data)))
        elif isinstance(obj._fields[field_name], BooleanField):
            return_data.append((field_name, bool(data)))
        elif isinstance(obj._fields[field_name], DateTimeField):
            return_data.append(field_name, datetime.datetime.strptime(data))
        else:
            return_data.append((field_name, data))
    return dict(return_data)


def to_confidence(value):
    min_value = torch.min(value)
    max_value = torch.max(value)
    return 1 - (value - min_value) / (max_value - min_value)
