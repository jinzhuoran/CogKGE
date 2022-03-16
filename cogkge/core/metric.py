import copy
import math
from collections import defaultdict

import numpy as np
import prettytable as pt
import torch
import torch.utils.data as Data
from tqdm import tqdm


class Link_Prediction(object):
    def __init__(self,
                 batch_size,
                 reverse,
                 node_lut=None,
                 relation_lut=None,
                 time_lut = None,
                 link_prediction_raw=True,
                 link_prediction_filt=False,
                 metric_pattern="score_based",
                 ):
        """
        验证器的参数设置

        :param batch_size: 验证器的batch_size
        :param reverse: mr指标反转
        :param link_prediction_raw: raw类型的链接预测
        :param link_prediction_filt: filt类型的连接预测
        :param metric_pattern: 选择score_based或者classification_based
        """
        self.batch_size = batch_size
        self.reverse = reverse
        self.node_lut = node_lut
        self.relation_lut = relation_lut
        self.time_lut = time_lut
        self.link_prediction_raw = link_prediction_raw
        self.link_prediction_filt = link_prediction_filt
        if metric_pattern not in ["classification_based", "score_based"]:
            raise ValueError(
                "Metric pattern {} is not supported.Use \"classification_based\" or \"score_based\" instead."
                    .format(metric_pattern))
        self.metric_pattern = metric_pattern

        self.device = None
        self.total_epoch = None
        self.metric_type = None
        self.node_dict_len = None
        self.model_name = None
        self.logger = None
        self.writer = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.correct_triplets_dict = None
        self._metric_result_list = list()

        self.mode = None

        self.empty_result = {
            "raw_rank": -1,
            "filt_rank": -1,
            "Raw_MR": -1,
            "Raw_MRR": -1,
            "Raw_Hits1": -1,
            "Raw_Hits3": -1,
            "Raw_Hits10": -1,
            "Filt_MR": -1,
            "Filt_MRR": -1,
            "Filt_Hits1": -1,
            "Filt_Hits3": -1,
            "Filt_Hits10": -1}

    def initialize(self,
                   device,
                   total_epoch,
                   metric_type,
                   node_dict_len,
                   model_name,
                   logger=None,
                   writer=None,
                   train_dataset=None,
                   valid_dataset=None,
                   test_dataset=None):
        """
        验证器的初始化

        :param device: 验证器的位置
        :param total_epoch: 训练的总轮数
        :param metric_type: 验证验证集或者测试集
        :param node_dict_len: 节点的字典长度
        :param model_name: 模型的名字
        :param logger: log
        :param writer: 可视化
        :param train_dataset: 训练集
        :param valid_dataset: 验证集
        :param test_dataset: 测试集
        """
        self.device = device
        self.total_epoch = total_epoch
        self.metric_type = metric_type
        self.node_dict_len = node_dict_len
        self.model_name = model_name
        self.logger = logger
        self.writer = writer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        if not train_dataset and not valid_dataset and not test_dataset:
            raise ValueError("Not one dataset is specified in metric!")

    def _create_correct_node_dict(self, dataset):
        print("Creating correct index...")
        if self.metric_pattern == "classification_based":
            hr_t_vocab = defaultdict(list)
            rt_h_vocab = defaultdict(list)

            for index in tqdm(range(len(dataset))):
                h,r,t = dataset.data[index]
                # h, r, t = dataset.data[0][index]
                hr_t_vocab[(h, r)].append(t)
                rt_h_vocab[(r, t)].append(h)
            node_dict = {"head": rt_h_vocab, "tail": hr_t_vocab}

        if self.metric_pattern == "score_based":
            node_dict = defaultdict(dict)
            for index in tqdm(range(len(dataset))):
                r_t = tuple(dataset.data[index][1:3])
                h_r = tuple(dataset.data[index][:2])
                h = torch.tensor(dataset.data[index][0])
                t = torch.tensor(dataset.data[index][2])
                if r_t not in node_dict["head"]:
                    node_dict["head"].setdefault(r_t, [])
                node_dict["head"][r_t].append(h)
                if h_r not in node_dict["tail"]:
                    node_dict["tail"].setdefault(h_r, [])
                node_dict["tail"][h_r].append(t)
        return node_dict

    def establish_correct_triplets_dict(self):
        """
        建立正确的三元组元素字典
        {(correct_head,correct_relation):correct_tail}
        {(correct_tail,correct_relation):correct_head}
        """
        if self.train_dataset != None:
            self._correct_train_node_dict = self._create_correct_node_dict(self.train_dataset)
        if self.valid_dataset != None:
            self._correct_valid_node_dict = self._create_correct_node_dict(self.valid_dataset)
        if self.test_dataset != None:
            self._correct_test_node_dict = self._create_correct_node_dict(self.test_dataset)

    def _calculate_rank_classification_based(self, single_sample):
        with torch.no_grad():
            data_batch = single_sample.to(self.device)
            e2_idx = data_batch[:, 2]
            predictions = self._model(data_batch)

            if self.link_prediction_raw:
                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()
                ranks = [np.where(sort_idxs[j] == e2_idx[j].item())[0][0] + 1 for j in range(data_batch.shape[0])]
                self._raw_rank_list.append(ranks)

            if self.link_prediction_filt:
                er_vocab = {}
                for word in {"train", "valid", "test"}:
                    name = "_correct_{}_node_dict".format(word)
                    if hasattr(self, name):
                        tmp_node_dict = getattr(self, name)
                        er_vocab.update(tmp_node_dict["tail"])

                for j in range(data_batch.shape[0]):
                    filt = er_vocab[(data_batch[j][0].item(), data_batch[j][1].item())]
                    target_value = predictions[j, e2_idx[j]].item()
                    predictions[j, filt] = 0.0
                    predictions[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()
                ranks = [np.where(sort_idxs[j] == e2_idx[j].item())[0][0] + 1 for j in range(data_batch.shape[0])]
                self._filt_rank_list.append(ranks)

    def _caculate_rank(self, single_sample, entity_type, data_dict):
        """
        计算batch的验证结果

        :param single_sample: 待验证的batch数据(1,3,self._outer_batch_size)
        :param entity_type: 替换的是head或者tail
        :param data_dict: {"h":tensor,"r":tensor","t":tensor,...}
        :return:
        """
        column_index_dict = {"head": 0, "tail": 2}
        single_triplet = copy.deepcopy(single_sample)  # (1,3,self._outer_batch_size)
        correct_id = single_triplet[:, column_index_dict[entity_type]]  # (1,self._outer_batch_size)
        expanded_triplet = single_triplet.expand(self.node_dict_len, 3,
                                                 self._single_batch_len)  # (self._node_dict_len,3,self._single_batch_len)
        node = torch.arange(0, self.node_dict_len).to(self.device)
        expanded_node = torch.unsqueeze(torch.unsqueeze(node, dim=1), dim=1)  # (self._node_dict_len,1,1)
        if entity_type == "head":
            original_r_t = expanded_triplet[:, 1:3, :]  # (self._node_dict_len,2,self._single_batch_len)
            expanded_node = expanded_node.expand(self.node_dict_len, 1,
                                                 self._single_batch_len)  # (self._node_dict_len,1,self._single_batch_len)
            expanded_triplet = torch.cat([expanded_node, original_r_t],
                                         dim=1)  # (self._node_dict_len,3,self._single_batch_len)
        if entity_type == "tail":
            original_h_r = expanded_triplet[:, 0:2, :]
            expanded_node = expanded_node.expand(self.node_dict_len, 1, self._single_batch_len)
            expanded_triplet = torch.cat([original_h_r, expanded_node], dim=1)

        expanded_triplet = expanded_triplet.permute(2, 0, 1)  # (self._single_batch_len,self._node_dict_len,3)
        expanded_triplet = torch.reshape(expanded_triplet, (-1, 3))  # (self._single_batch_len * self._node_dict_len,3)

        score_list = list()
        for i in range(math.ceil(self.node_dict_len / self.batch_size)):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, self.node_dict_len * self._single_batch_len)
            data_input = expanded_triplet[start:end, :]
            data_input = self.tensor_to_tuple(data_input,data_dict)
            with torch.no_grad():
                result = self._model(data_input)
                score_list.append(result)
        sorted_score_rank = torch.argsort(torch.cat(score_list, dim=0).reshape(self._single_batch_len, -1).T,
                                          dim=0)  # torch.Size([14541, batch])
        len_b = len(sorted_score_rank)
        correct_triplet_rank = torch.mm(torch.unsqueeze(torch.arange(0, len_b), dim=0),
                                        (sorted_score_rank == correct_id).type(torch.LongTensor))[0] + 1

        if self.link_prediction_raw:
            raw_correct_triplet_rank = self.node_dict_len + 1 - correct_triplet_rank if self.reverse else correct_triplet_rank
            self._raw_rank_list.append(raw_correct_triplet_rank.tolist())

        if self.link_prediction_filt:
            arr = 1 if entity_type == "head" else 0
            sorted_score_rank = sorted_score_rank.cpu()
            triplet_key_list = list(tuple(single_triplet[0][arr:arr + 2, :].permute(1, 0).cpu().numpy()[i]) for i in
                                    range(len(single_triplet[0][arr:arr + 2, :].permute(1, 0))))
            for i, triplet_key in enumerate(triplet_key_list):
                valid_correct_node_index_list = []
                train_correct_node_index_list = []
                test_correct_node_index_list = []
                if triplet_key in self._correct_train_node_dict[entity_type].keys():
                    valid_correct_node_index_list = torch.where(sorted_score_rank[:, i] == torch.unsqueeze(
                        torch.tensor(self._correct_train_node_dict[entity_type][triplet_key]), dim=1))[1] + 1
                if triplet_key in self._correct_valid_node_dict[entity_type].keys():
                    train_correct_node_index_list = torch.where(sorted_score_rank[:, i] == torch.unsqueeze(
                        torch.tensor(self._correct_valid_node_dict[entity_type][triplet_key]), dim=1))[1] + 1
                if self.test_dataset != None:
                    if triplet_key in self._correct_test_node_dict[entity_type].keys():
                        test_correct_node_index_list = torch.where(sorted_score_rank[:, i] == torch.unsqueeze(
                            torch.tensor(self._correct_test_node_dict[entity_type][triplet_key]), dim=1))[1] + 1
                valid_correct_before_num = torch.sum(
                    torch.as_tensor(valid_correct_node_index_list) < correct_triplet_rank[i])
                train_correct_before_num = torch.sum(
                    torch.as_tensor(train_correct_node_index_list) < correct_triplet_rank[i])
                if self.test_dataset != None:
                    test_correct_before_num = torch.sum(
                        torch.as_tensor(test_correct_node_index_list) < correct_triplet_rank[i])
                else:
                    test_correct_before_num = 0
                correct_triplet_num = valid_correct_before_num + train_correct_before_num + test_correct_before_num
                filt_correct_triplet_rank = correct_triplet_rank[i] - correct_triplet_num
                self._filt_rank_list.append(filt_correct_triplet_rank)
    def get_batch(self,data_batch):
        return torch.cat([torch.unsqueeze(data_batch[index],dim=1) for index in ["h","r","t"]],dim=1) # (batch,3)

    def record_mode(self,test_sample):
        """
        test_sample:(tensor_h,tensor_r,tensor_t,...)
        """
        len2mode = {3:"normal",
                    5:"time",
                    6:"type",
                    7:"description"}
        self.mode = len2mode[len(test_sample)]

    def tensor_to_tuple(self,data_tensor,data_dict):
        """
        data_tensor:  tensor(batch_size,3|5|6|7)
        return: (tensor(batch_size,),tensor(batch_size,),tensor(batch_size,),...)
        """
        sample = {"h": data_tensor[:,0],
                  "r": data_tensor[:,1],
                  "t": data_tensor[:,2]}
        return self.update_sample(sample,data_dict)

    def update_sample(self,sample,data_dict):
        if self.mode == "type":
            sample.update({"h_type":self.node_lut.type[list(sample["h"])],
                           "t_type":self.node_lut.type[list(sample["t"])],
                           "r_type":self.relation_lut.type[list(sample["r"])]})
        elif self.mode == "description":
            sample.update({"h_token": self.node_lut.token[list(sample["h"])],
                           "t_token": self.node_lut.token[list(sample["t"])],
                           "h_mask": self.node_lut.mask[list(sample["h"])],
                           "t_mask": self.node_lut.mask[list(sample["t"])]})
        elif self.mode == "time":
            sample.update({"start":self.expand_tensor(data_dict["start"]),
                           "end":self.expand_tensor(data_dict["end"])})
        elif self.mode == "normal":
            pass
        else:
            raise ValueError("Mode {} is not defined!".format(self.mode))
        return list(sample.values())

    def expand_tensor(self,target):
        """
        target: tensor(self._single_batch_len,)
        return: tensor(self._single_batch_len * self.node_dict_len,)
        """
        return torch.flatten(target.expand(self.node_dict_len , self._single_batch_len).T)

    def tuple_to_dict(self,data_tuple):
        data_dict = {
            "h":data_tuple[0],
            "r":data_tuple[1],
            "t":data_tuple[2],
        }
        if len(data_tuple) == 3:
            pass
        elif len(data_tuple) == 5: # time info
            data_dict.update({
                "start":data_tuple[3],
                "end":data_tuple[4],
            })
        elif len(data_tuple) == 6: # type info
            data_dict.update({
                "h_type":data_tuple[3],
                "t_type":data_tuple[4],
                "r_type":data_tuple[5],
            })
        elif len(data_tuple) == 7: # descriptions info
            data_dict.update({
                "h_token":data_tuple[3],
                "t_token":data_tuple[4],
                "h_mask":data_tuple[5],
                "t_mask":data_tuple[6],
            })
        else:
            raise ValueError("Length of data_tuple {} unexpected!".format(len(data_tuple)))
        return data_dict

    def caculate(self, model, current_epoch):
        """
        验证当前模型

        :param model: 待验证的模型
        :param current_epoch: 当前轮数
        """

        torch.cuda.empty_cache()
        self._model = model
        self._current_epoch = current_epoch
        self._current_result = self.empty_result.copy()
        self._raw_rank_list = list()
        self._filt_rank_list = list()
        self._outer_batch_size = max(math.floor(self.batch_size / self.node_dict_len), 1)

        if self.metric_type == "valid":
            metric_dataset = self.valid_dataset
        elif self.metric_type == "test":
            metric_dataset = self.test_dataset
        else:
            raise ValueError("Please choose correct metric_type:valid or test!")

        metric_loader = Data.DataLoader(dataset=metric_dataset,
                                        batch_size=self._outer_batch_size,
                                        shuffle=False)
        test_sample = next(iter(metric_loader))
        self.record_mode(test_sample)
        for step, single_sample in enumerate(tqdm(metric_loader)):
            data_dict = self.tuple_to_dict(single_sample)
            single_sample = self.get_batch(data_dict)
            if self.metric_pattern == "score_based":
                self._single_batch_len = single_sample.shape[0] # (self._outer_batch_size,3)
                single_sample = single_sample.permute(1, 0)  # (3,self._outer_batch_size)
                single_sample = torch.unsqueeze(single_sample, dim=0)  # (1,3,self._outer_batch_size)
                single_sample = single_sample.to(self.device)
                self._caculate_rank(single_sample, "tail",data_dict)
                self._caculate_rank(single_sample, "head",data_dict)
                # if step>3:
                #     break
            if self.metric_pattern == "classification_based":
                self._calculate_rank_classification_based(single_sample)

        if self.link_prediction_raw:
            self._raw_rank_list = sum(self._raw_rank_list, [])
            current_raw_rank = torch.tensor(self._raw_rank_list, dtype=torch.float64)
            self._current_result["Raw_Hits1"] = round(
                (torch.sum((current_raw_rank <= 1)) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._current_result["Raw_Hits3"] = round(
                (torch.sum(current_raw_rank <= 3) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._current_result["Raw_Hits10"] = round(
                (torch.sum(current_raw_rank <= 10) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._current_result["Raw_MR"] = round(torch.mean(current_raw_rank).item(), 3)
            self._current_result["Raw_MRR"] = round(torch.mean(1 / current_raw_rank).item(), 3)
            # print(current_raw_rank)
        if self.link_prediction_filt:
            if self.metric_pattern == 'classification_based':
                # expand the nested list
                self._filt_rank_list = sum(self._filt_rank_list, [])
            current_filt_rank = torch.tensor(self._filt_rank_list, dtype=torch.float64)
            self._current_result["Filt_Hits1"] = round(
                (torch.sum((current_filt_rank <= 1)) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._current_result["Filt_Hits3"] = round(
                (torch.sum(current_filt_rank <= 3) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._current_result["Filt_Hits10"] = round(
                (torch.sum(current_filt_rank <= 10) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._current_result["Filt_MR"] = round(torch.mean(current_filt_rank).item(), 3)
            self._current_result["Filt_MRR"] = round(torch.mean(1 / current_filt_rank).item(), 3)
        if self.link_prediction_raw or self.link_prediction_filt:
            self._metric_result_list.append([self._current_epoch,
                                             self._current_result["Raw_Hits1"],
                                             self._current_result["Raw_Hits3"],
                                             self._current_result["Raw_Hits10"],
                                             self._current_result["Raw_MR"],
                                             self._current_result["Raw_MRR"],
                                             self._current_result["Filt_Hits1"],
                                             self._current_result["Filt_Hits3"],
                                             self._current_result["Filt_Hits10"],
                                             self._current_result["Filt_MR"],
                                             self._current_result["Filt_MRR"],
                                             ])

    def get_Raw_MR(self):
        return self._current_result["Raw_MR"]

    def get_Raw_MRR(self):
        return self._current_result["Raw_MRR"]

    def get_Raw_Hits1(self):
        return self._current_result["Raw_Hits1"]

    def get_Raw_Hits3(self):
        return self._current_result["Raw_Hits3"]

    def get_Raw_Hits10(self):
        return self._current_result["Raw_Hits10"]

    def get_Filt_MR(self):
        return self._current_result["Filt_MR"]

    def get_Filt_MRR(self):
        return self._current_result["Filt_MRR"]

    def get_Filt_Hits1(self):
        return self._current_result["Filt_Hits1"]

    def get_Filt_Hits3(self):
        return self._current_result["Filt_Hits3"]

    def get_Filt_Hits10(self):
        return self._current_result["Filt_Hits10"]

    def print_current_table(self):
        tb = pt.PrettyTable()
        tb.field_names = [self.model_name,
                          "Data",
                          "Epoch/Total",
                          "Hits@1",
                          "Hits@3",
                          "Hits@10",
                          "MR",
                          "MRR"]
        if self.link_prediction_raw:
            tb.add_row(["Raw",
                        self.metric_type,
                        str(self._current_epoch) + "/" + str(self.total_epoch),
                        self._current_result["Raw_Hits1"],
                        self._current_result["Raw_Hits3"],
                        self._current_result["Raw_Hits10"],
                        self._current_result["Raw_MR"],
                        self._current_result["Raw_MRR"]])
        if self.link_prediction_filt:
            tb.add_row(["Filt",
                        self.metric_type,
                        str(self._current_epoch) + "/" + str(self.total_epoch),
                        self._current_result["Filt_Hits1"],
                        self._current_result["Filt_Hits3"],
                        self._current_result["Filt_Hits10"],
                        self._current_result["Filt_MR"],
                        self._current_result["Filt_MRR"]])
        print(tb)

    def print_best_table(self, front=3, key="Filt_Hits@10"):
        front = len(self._metric_result_list) if len(self._metric_result_list) < front else front
        strat_index = 5 if self.link_prediction_raw and self.link_prediction_filt else 0
        type_dict = {"Raw_Hits@1": [1, True],
                     "Raw_Hits@3": [2, True],
                     "Raw_Hits@10": [3, True],
                     "Raw_MR": [4, False],
                     "Raw_MRR": [5, True],
                     "Filt_Hits@1": [1 + strat_index, True],
                     "Filt_Hits@3": [2 + strat_index, True],
                     "Filt_Hits@10": [3 + strat_index, True],
                     "Filt_MR": [4 + strat_index, False],
                     "Filt_MRR": [5 + strat_index, True]}
        table_title = [self.model_name,
                       "Last",
                       "Best",
                       "2nd",
                       "3rd",
                       "4th",
                       "5th"]
        tb = pt.PrettyTable()
        tb.field_names = table_title[:front + 2]
        last_result = self._metric_result_list[-1]
        self._metric_result_list.sort(key=lambda x: x[type_dict[key][0]], reverse=type_dict[key][1])
        self._metric_result_list = [last_result] + self._metric_result_list
        result_list_T = np.array(self._metric_result_list).T.tolist()
        table_row_title = list()
        raw_table_row_title = list()
        filt_table_row_title = list()
        if self.link_prediction_raw:
            raw_table_row_title = ["Raw_Hits@1", "Raw_Hits@3", "Raw_Hits@10", "Raw_MR", "Raw_MRR"]
        if self.link_prediction_filt:
            filt_table_row_title = ["Filt_Hits@1", "Filt_Hits@3", "Filt_Hits@10", "Filt_MR", "Filt_MRR"]
        if self.link_prediction_raw or self.link_prediction_filt:
            table_row_title = ["Epoch"] + raw_table_row_title + filt_table_row_title
        for i in range(len(table_row_title)):
            tb.add_row([table_row_title[i]] + result_list_T[i][:front + 1])
        self.logger.info("\n")
        self.logger.info(tb)
        # if self.link_prediction_raw:
        #     self.logger.info(tb)
        # self.logger.info("Best: Epoch {}  Raw_Hits@1:{}   Raw_Hits@3:{}   Raw_Hits@10:{}   Raw_MR:{}   Raw_MRR:{}".format(
        #     self._metric_result_list[1][0],
        #     self._metric_result_list[1][1],
        #     self._metric_result_list[1][2],
        #     self._metric_result_list[1][3],
        #     self._metric_result_list[1][4],
        #     self._metric_result_list[1][5]))
        # if self.link_prediction_filt:
        #     self.logger.info(tb)
        # self.logger.info("Best: Epoch {}  Filt_Hits@1:{}   Filt_Hits@3:{}   Filt_Hits@10:{}   Filt_MR:{}   Filt_MRR:{}   ".format(
        #     self._metric_result_list[1][0],
        #     self._metric_result_list[1][1+strat_index],
        #     self._metric_result_list[1][2+strat_index],
        #     self._metric_result_list[1][3+strat_index],
        #     self._metric_result_list[1][4+strat_index],
        #     self._metric_result_list[1][5+strat_index]))

    def log(self):
        if self.link_prediction_raw:
            self.logger.info(
                "{} Epoch {}/{}  Raw_Hits@1:{}   Raw_Hits@3:{}   Raw_Hits@10:{}   Raw_MR:{}   Raw_MRR:{}".format(
                    self.metric_type, self._current_epoch, self.total_epoch, self._current_result["Raw_Hits1"],
                    self._current_result["Raw_Hits3"], self._current_result["Raw_Hits10"],
                    self._current_result["Raw_MR"], self._current_result["Raw_MRR"]))
        if self.link_prediction_filt:
            self.logger.info(
                "{} Epoch {}/{}  Filt_Hits@1:{}   Filt_Hits@3:{}   Filt_Hits@10:{}   Filt_MR:{}   Filt_MRR:{}   ".format(
                    self.metric_type, self._current_epoch, self.total_epoch, self._current_result["Filt_Hits1"],
                    self._current_result["Filt_Hits3"], self._current_result["Filt_Hits10"],
                    self._current_result["Filt_MR"], self._current_result["Filt_MRR"]))

    def write(self):
        if self.writer != None:
            if self.link_prediction_raw:
                self.writer.add_scalars("Hits@1",
                                        {"{}_Raw_Hits@1".format(self.metric_type): self._current_result["Raw_Hits1"]},
                                        self._current_epoch)
                self.writer.add_scalars("Hits@3",
                                        {"{}_Raw_Hits@3".format(self.metric_type): self._current_result["Raw_Hits3"]},
                                        self._current_epoch)
                self.writer.add_scalars("Hits@10",
                                        {"{}_Raw_Hits@10".format(self.metric_type): self._current_result["Raw_Hits10"]},
                                        self._current_epoch)
                self.writer.add_scalars("MR", {"{}_Raw_MR".format(self.metric_type): self._current_result["Raw_MR"]},
                                        self._current_epoch)
                self.writer.add_scalars("MRR", {"{}_Raw_MRR".format(self.metric_type): self._current_result["Raw_MRR"]},
                                        self._current_epoch)
            if self.link_prediction_filt:
                self.writer.add_scalars("Hits@1",
                                        {"{}_Filt_Hits@1".format(self.metric_type): self._current_result["Filt_Hits1"]},
                                        self._current_epoch)
                self.writer.add_scalars("Hits@3",
                                        {"{}_Filt_Hits@3".format(self.metric_type): self._current_result["Filt_Hits3"]},
                                        self._current_epoch)
                self.writer.add_scalars("Hits@10", {
                    "{}_Filt_Hits@10".format(self.metric_type): self._current_result["Filt_Hits10"]},
                                        self._current_epoch)
                self.writer.add_scalars("MR", {"{}_Filt_MR".format(self.metric_type): self._current_result["Filt_MR"]},
                                        self._current_epoch)
                self.writer.add_scalars("MRR",
                                        {"{}_Filt_MRR".format(self.metric_type): self._current_result["Filt_MRR"]},
                                        self._current_epoch)
