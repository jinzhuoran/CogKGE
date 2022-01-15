import os
import json
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

class Kr_Predictior:
    def __init__(self,
                 model,
                 pretrained_model_path,
                 device,
                 node_lut,
                 relation_lut,
                 reprocess=True,
                 top=10,
                 name2id_node_dict_path="data/name2id_node_dict.json",
                 name2id_relation_dict_path="data/name2id_relation_dict.json"):
        """
        三元组链接预测

        :param model:模型
        :param pretrained_model_path: 知识表示预训练模型路径
        :param device:显卡编号
        :param node_lut:节点查找表
        :param relation_lut:关系查找表
        :param reprocess:重新处理
        :param top:查找前top个结果
        :param name2id_node_dict_path:名字转id的字典
        :param name2id_relation_dict_path:关系转id的字典
        """

        if not isinstance(reprocess, bool):
            raise TypeError("param reprocess is True or False!")

        self.model=model
        self.pretrained_mode_path=pretrained_model_path
        self.device=device
        self.node_lut=node_lut
        self.relation_lut=relation_lut
        self.reprocess=reprocess
        self.top=top
        self.name2id_node_dict_path =name2id_node_dict_path
        self.name2id_relation_dict_path =name2id_relation_dict_path

        if os.path.exists(self.pretrained_mode_path):
            self.model.load_state_dict(torch.load(self.pretrained_mode_path))
            self.model=self.model.to(self.device)
        else:
            raise FileExistsError("pretrained_model_path doesn't exist!")

        self.node_len = len(self.node_lut)
        self.relation_len=len(self.relation_lut)
        self.all_node_index_column_matrix = torch.unsqueeze(torch.arange(self.node_len).to(self.device),dim=1)
        self.all_node_index_row_vector = torch.arange(self.node_len).to(self.device)
        self.all_node_embedding = self.model.entity_embedding_base(self.all_node_index_row_vector)
        self.all_relation_index_column_matrix=torch.unsqueeze(torch.arange(self.relation_len).to(self.device),dim=1)
        # self.all_relation_index_row_vector = torch.arange(self.relation_len).to(self.device)
        # self.all_relation_embedding = self.model.relation_embedding_base(self.all_relation_index_row_vector)

        self.name2id_node_dict=defaultdict(list)
        self.name2id_relation_dict = defaultdict(list)
        self._create_name2id_node_dict()
        self._create_name2id_relation_dict()

    def _create_name2id_node_dict(self):
        """
        建立节点名字到id的查找表
        """
        if self.reprocess:
            for i in tqdm(range(self.node_len)):
                item_df=self.node_lut[i]
                item={}
                item["id"]=str(item_df["name_id"])
                item["name"]=item_df["name"]
                item["summary"]=item_df["description"]
                self.name2id_node_dict[item_df["name"]].append(item)
            json.dump(self.name2id_node_dict,open(self.name2id_node_dict_path,"w"),indent=4,sort_keys=False)
        else:
            if os.path.exists(self.name2id_node_dict_path):
                with open(self.name2id_node_dict_path) as file:
                    self.name2id_node_dict=json.load(file)
            else:
                raise FileExistsError("{} does not exist!".format(self.name2id_node_dict_path))

    def _create_name2id_relation_dict(self):
        """
        建立关系名字到id的查找表
        """
        if self.reprocess:
            for i in tqdm(range(self.relation_len)):
                item_df = self.relation_lut[i]
                item = {}
                item["id"] = str(item_df["name_id"])
                item["name"] = item_df["name"]
                self.name2id_relation_dict[item_df["name"]].append(item)
            json.dump(self.name2id_relation_dict, open(self.name2id_relation_dict_path, "w"), indent=4, sort_keys=True)
        else:
            if os.path.exists(self.name2id_relation_dict_path):
                with open(self.name2id_relation_dict_path) as file:
                    self.name2id_relation_dict = json.load(file)
            else:
                raise FileExistsError("{} does not exist!".format(self.name2id_relation_dict_path))

    def _caculate_triplets_topk_node(self,triplet_id_matric):
        """
        计算三元组得分最高的k个头节点或者尾节点

        :param triplet_id_matric: 更换头结点和尾节点的三元组
        :return: 得分topk的三元组列表
        """
        distance = self.model(triplet_id_matric)
        value, index = torch.topk(distance, self.top, largest=False)
        node_list = []
        for i in range(self.top):
            id = int(index[i])
            item = {}
            item["name_id"] = self.node_lut[id]["name_id"]
            item["name"] = self.node_lut[id]["name"]
            item["name_rdf"] = self.node_lut[id]["name_rdf"]
            item["type"] = self.node_lut[id]["type"]
            item["type_rdf"] = self.node_lut[id]["type_rdf"]
            item["node_type"] = self.node_lut[id]["node_type"]
            item["description"] = self.node_lut[id]["description"]
            item["confidence"] = value[i].item()
            node_list.append(item)
        return node_list


    def fuzzy_query_node_keyword(self,node_keyword=None):
        """
        模糊查询节点名字

        :param node_keyword: 输入节点名字
        :return: 模糊节点列表
        """
        if node_keyword is None:
            return [self.name2id_node_dict["Copa Colombia"][0],
                    self.name2id_node_dict["Speed skating at the 2018 Winter Olympics – Women's 500 metres"][0],
                    self.name2id_node_dict["Swimming at the 1988 Summer Olympics – Women's 50 metre freestyle"][0],
                    self.name2id_node_dict["2015 Malta Badminton Championships"][0],
                    self.name2id_node_dict["Syracuse Grand Prix"][0],
                    self.name2id_node_dict["Western Canada Hockey League"][0],
                    self.name2id_node_dict["Anke Engelke"][0],
                    self.name2id_node_dict["Aviram Baruchyan"][0],
                    self.name2id_node_dict["Attica Prison riot"][0],
                    self.name2id_node_dict["Wayne Odesnik"][0]]
        else:
            return self.name2id_node_dict[node_keyword]

    def fuzzy_query_relation_keyword(self,relation_keyword=None):
        """
        模糊查询关系名字

        :param relation_keyword: 输入关系名字
        :return: 模糊关系列表
        """
        if relation_keyword is None:
            return [self.name2id_relation_dict["hasWonPrize"][0],
                    self.name2id_relation_dict["nextEvent"][0],
                    self.name2id_relation_dict["participant of"][0],
                    self.name2id_relation_dict["previousEvent"][0],
                    self.name2id_relation_dict["award received"][0],
                    self.name2id_relation_dict["author"][0],
                    self.name2id_relation_dict["has cause"][0],
                    self.name2id_relation_dict["similar"][0],
                    self.name2id_relation_dict["represents"][0],
                    self.name2id_relation_dict["style"][0]]
        return self.name2id_relation_dict[relation_keyword]


    def predict_similar_node(self,node_id):
        """
        预测相似的节点

        :param node_id: 输入节点id
        :return: similar_node_list 相似节点列表
        """

        node_embedding = torch.unsqueeze(self.model.entity_embedding_base(torch.tensor(node_id).to(self.device)),dim=0)
        distance = F.pairwise_distance(node_embedding, self.all_node_embedding, p=2)
        value, index = torch.topk(distance,self.top+1, largest=False)
        value=value[1:]
        index=index[1:]
        similar_node_list=[]
        for i in range(self.top):
            id=int(index[i])
            item={}
            item["name_id"]=self.node_lut[id]["name_id"]
            item["name"]=self.node_lut[id]["name"]
            item["name_rdf"]=self.node_lut[id]["name_rdf"]
            item["type"]=self.node_lut[id]["type"]
            item["type_rdf"]=self.node_lut[id]["type_rdf"]
            item["node_type"]=self.node_lut[id]["node_type"]
            item["description"]=self.node_lut[id]["description"]
            item["confidence"]=value[i].item()
            similar_node_list.append(item)

        return similar_node_list

    def predcit_tail(self,head_id,relation_id=None):
        """
        根据头节点和关系预测尾节点
        如果关系为空，则遍历所有关系，计算出每种关系得分最高的，选出前topk个节点

        :param head_id: 输入头节点id
        :param relation_id: 输入关系id
        :return: tail_list 尾实体列表
        """
        if relation_id is None:
            with torch.no_grad():
                score_list=[] #每一个位置i对应关系i最好的得分
                tail_index_list=[] #每一个位置i对应关系i最好的尾节点id
                for relation_id in range(self.relation_len):
                    head_id_vector = torch.tensor(head_id).expand(self.node_len, 1).to(self.device)
                    relation_id_vector = torch.tensor(relation_id).expand(self.node_len, 1).to(self.device)
                    tail_id_vectoe = self.all_node_index_column_matrix.clone()
                    triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vectoe], dim=1)
                    distance = self.model(triplet_id_matric)
                    tail_index = distance.argmin(dim=0)#关系i score最好的 尾节点id
                    score = distance[tail_index]#关系i score最好的 得分数值
                    score_list.append(score)
                    tail_index_list.append(tail_index)
                score = torch.tensor(score_list)
                tail_index = torch.tensor(tail_index_list)
                value, relation_index = torch.topk(score, self.top, largest=False)
                tail_list = []
                for i in range(self.top):
                    tail_id = int(tail_index[i])
                    relation_id=int(relation_index[i])
                    item = {}
                    item["name_id"] = self.node_lut[tail_id]["name_id"]
                    item["name"] = self.node_lut[tail_id]["name"]
                    item["name_rdf"] = self.node_lut[tail_id]["name_rdf"]
                    item["type"] = self.node_lut[tail_id]["type"]
                    item["type_rdf"] = self.node_lut[tail_id]["type_rdf"]
                    item["node_type"] = self.node_lut[tail_id]["node_type"]
                    item["description"] = self.node_lut[tail_id]["description"]
                    item["confidence"] = value[i].item()
                    item["triplet_id"]=(head_id,relation_id,tail_id)
                    item["triplet_name"] = (self.node_lut[head_id]["name"],self.relation_lut[relation_id]["name"],self.node_lut[tail_id]["name"])
                    tail_list.append(item)
                return tail_list
        else:
            head_id_vector = torch.tensor(head_id).expand(self.node_len, 1).to(self.device)
            relation_id_vector = torch.tensor(relation_id).expand(self.node_len, 1).to(self.device)
            tail_id_vectoe = self.all_node_index_column_matrix.clone()
            triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vectoe], dim=1)
            tail_list = self._caculate_triplets_topk_node(triplet_id_matric)
            return tail_list


    def predict_head(self,tail_id,relation_id=None):
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
                value, relation_index = torch.topk(score, self.top, largest=False)
                head_list = []
                for i in range(self.top):
                    head_id = int(head_index[i])
                    relation_id = int(relation_index[i])
                    item = {}
                    item["name_id"] = self.node_lut[head_id]["name_id"]
                    item["name"] = self.node_lut[head_id]["name"]
                    item["name_rdf"] = self.node_lut[head_id]["name_rdf"]
                    item["type"] = self.node_lut[head_id]["type"]
                    item["type_rdf"] = self.node_lut[head_id]["type_rdf"]
                    item["node_type"] = self.node_lut[head_id]["node_type"]
                    item["description"] = self.node_lut[head_id]["description"]
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
            head_list = self._caculate_triplets_topk_node(triplet_id_matric)
            return head_list



    def predict_relation(self,head_id,tail_id):
        """
        根据头节点和尾节点预测关系

        :param head_id: 输入头节点id
        :param tail_id: 输入尾节点id
        :return: relation_list 关系列表
        """

        head_id_vector = torch.tensor(head_id).expand(self.relation_len, 1).to(self.device)
        relation_id_vector=self.all_relation_index_column_matrix.clone()
        tail_id_vector = torch.tensor(tail_id).expand(self.relation_len, 1).to(self.device)
        triplet_id_matric = torch.cat([head_id_vector, relation_id_vector, tail_id_vector], dim=1)
        distance = self.model(triplet_id_matric)
        value, index = torch.topk(distance, self.top, largest=False)
        relation_list = []
        for i in range(self.top):
            id = int(index[i])
            item = {}
            item["name_id"] = self.relation_lut[id]["name_id"]
            item["name"] = self.relation_lut[id]["name"]
            item["name_rdf"] = self.relation_lut[id]["rdf"]
            item["confidence"] = value[i].item()
            relation_list.append(item)

        return relation_list



