import os
import re

import prettytable as pt
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from cogkge.core import DataLoaderX
from .log import save_logger
from ..utils.kr_utils import cal_output_path


class Evaluator(object):
    def __init__(self,
                 test_dataset,
                 test_sampler,
                 evaluator_batch_size,
                 model,
                 metric,
                 device,
                 output_path,
                 trained_model_path=None,
                 train_dataset=None,
                 valid_dataset=None,
                 lookuptable_E=None,
                 lookuptable_R=None,
                 log=None,
                 dataloaderX=False,
                 num_workers=None,
                 pin_memory=False,
                 ):
        self.test_dataset = test_dataset
        self.test_sampler = test_sampler
        self.evaluator_batch_size = evaluator_batch_size
        self.model = model
        self.trained_model_path = trained_model_path
        self.metric = metric
        self.device = device
        self.output_path = output_path
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lookuptable_E = lookuptable_E
        self.lookuptable_R = lookuptable_R
        self.log = log
        self.dataloaderX = dataloaderX
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_name = test_dataset.data_name

        # Set output_path
        output_path = os.path.join(output_path, "kr", self.data_name)
        self.output_path = cal_output_path(output_path, self.model.name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Set logger
        if log:
            logger = save_logger(os.path.join(self.output_path, "evaluator_run.log"))
            logger.info("Data Experiment Output Path:{}".format(self.output_path))
            self.logger = logger

        # Load Data
        if self.dataloaderX:
            self.test_loader = DataLoaderX(dataset=self.test_dataset, sampler=self.test_sampler,
                                           batch_size=self.evaluator_batch_size, num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)
        else:
            self.test_loader = Data.DataLoader(dataset=self.test_dataset, sampler=self.test_sampler,
                                               batch_size=self.evaluator_batch_size, num_workers=self.num_workers,
                                               pin_memory=self.pin_memory)


        # Load Trained Model
        self.trained_epoch = 0
        if self.trained_model_path:
            if os.path.exists(self.trained_model_path):
                string = self.trained_model_path
                pattern = r"^.*?/checkpoints/.*?_(.*?)epochs$"
                match = re.search(pattern, string)
                self.trained_epoch = int(match.group(1))
                self.model.load_state_dict(torch.load(os.path.join(self.trained_model_path, "Model.pkl")))
            else:
                raise FileExistsError("Trained_model_path doesn't exist!")

        # Load Dataparallel Training
        print("Available cuda devices:", torch.cuda.device_count())
        self.parallel_model = torch.nn.DataParallel(self.model)
        self.parallel_model = self.parallel_model.to(self.device)

        # Load Metric
        if self.metric:
            self.metric.initialize(device=self.device,
                                   total_epoch=0,
                                   metric_type="test",
                                   node_dict_len=len(self.lookuptable_E),
                                   model_name=self.model.name,
                                   logger=self.logger,
                                   writer=None,
                                   train_dataset=self.train_dataset,
                                   valid_dataset=self.valid_dataset,
                                   test_dataset=self.test_dataset)
            if self.metric.link_prediction_filt:
                self.metric.establish_correct_triplets_dict()

    def evaluate(self):
        current_epoch = self.trained_epoch
        print("Evaluating Model {} on Test Dataset...".format(self.model.name))
        self.metric.caculate(model=self.parallel_model, current_epoch=current_epoch)
        self.metric.print_current_table()
        self.metric.log()
        self.metric.write()

    def search_similar_entity(self, entity, top):
        print("Search_similar_entity Process...")
        entity_index = self.lookuptable_E.word2idx[entity]
        if self.model.name == "BoxE":
            entity_embedding = torch.unsqueeze(
                self.model.entity_embedding_base(torch.tensor(entity_index).to(self.device)), dim=0)
            total_index = torch.arange(len(self.lookuptable_E))
            total_embedding = self.model.entity_embedding_base(total_index.to(self.device))
        else:
            entity_embedding = torch.unsqueeze(self.model.entity_embedding(torch.tensor(entity_index).to(self.device)),
                                               dim=0)
            total_index = torch.arange(len(self.lookuptable_E))
            total_embedding = self.model.entity_embedding(total_index.to(self.device))
        distance = F.pairwise_distance(entity_embedding, total_embedding, p=2)
        value, index = torch.topk(distance, top + 1, largest=False)
        tb = pt.PrettyTable()
        tb.field_names = ["Query_Entity", "Rank", "Candidates", "Distance"]
        for i in range(len(value) - 1):
            tb.add_row([entity, i + 1, self.lookuptable_E.idx2word[index[i + 1].item()], round(value[i + 1].item(), 3)])
        print(tb)

    def search_similar_head(self, tail, top, relation=None, filt=True):
        print("Search_similar_head Process...")
        if relation:
            tail_index = torch.tensor(self.lookuptable_E.word2idx[tail]).expand(len(self.lookuptable_E), 1)
            relation_index = torch.tensor(self.lookuptable_R.word2idx[relation]).expand(len(self.lookuptable_E), 1)
            total_index = torch.unsqueeze(torch.arange(len(self.lookuptable_E)), dim=1)
            data_index = torch.cat([total_index, relation_index, tail_index], dim=1)
            distance = self.model(data_index.to(self.device))
            value, index = torch.topk(distance, top, largest=False)
            tb = pt.PrettyTable()
            tb.field_names = ["Query_Relation", "Query_Tail", "Rank", "Candidates_Head", "Distance"]
            for i in range(len(value)):
                tb.add_row(
                    [relation, tail, i + 1, self.lookuptable_E.idx2word[index[i].item()], round(value[i].item(), 3)])
            print(tb)
        else:
            score_list = list()
            head_list = list()
            tail_idx = self.lookuptable_E.word2idx[tail]
            with torch.no_grad():
                for relation_index in list(self.lookuptable_R.word2idx.values()):
                    tail_index = torch.tensor(self.lookuptable_E.word2idx[tail]).expand(len(self.lookuptable_E), 1)
                    relation_index = torch.tensor(relation_index).expand(len(self.lookuptable_E), 1)
                    total_index = torch.unsqueeze(torch.arange(len(self.lookuptable_E)), dim=1)
                    data_index = torch.cat([total_index, relation_index, tail_index], dim=1)
                    distance = self.model(data_index.to(self.device))
                    index = distance.argmin(dim=0)
                    value = distance[index]
                    score_list.append(value)
                    head_list.append(index)
                    pass
            score = torch.tensor(score_list)
            head = torch.tensor(head_list)
            value, index = torch.topk(score, top, largest=False)
            tb = pt.PrettyTable()
            tb.field_names = ["Query_Tail", "Rank", "Candidates_Head", "Candidates_Relation", "Distance"]
            for i in range(len(value)):
                tb.add_row([tail, i + 1, self.lookuptable_E.idx2word[head[index][i].item()],
                            self.lookuptable_R.idx2word[index[i].item()], round(value[i].item(), 3)])
            print(tb)

    def search_similar_tail(self, head, top, relation=None, filt=True):
        print("Search_similar_tail Process...")
        if relation:
            head_index = torch.tensor(self.lookuptable_E.word2idx[head]).expand(len(self.lookuptable_E), 1)
            relation_index = torch.tensor(self.lookuptable_R.word2idx[relation]).expand(len(self.lookuptable_E), 1)
            total_index = torch.unsqueeze(torch.arange(len(self.lookuptable_E)), dim=1)
            data_index = torch.cat([head_index, relation_index, total_index], dim=1)
            distance = self.model(data_index.to(self.device))
            value, index = torch.topk(distance, top, largest=False)
            tb = pt.PrettyTable()
            tb.field_names = ["Query_Head", "Query_Relation", "Rank", "Candidates_Tail", "Distance"]
            for i in range(len(value)):
                tb.add_row(
                    [head, relation, i + 1, self.lookuptable_E.idx2word[index[i].item()], round(value[i].item(), 3)])
            print(tb)
        else:
            score_list = list()
            tail_list = list()
            head_idx = self.lookuptable_E.word2idx[head]
            with torch.no_grad():
                for relation_index in list(self.lookuptable_R.word2idx.values()):
                    head_index = torch.tensor(self.lookuptable_E.word2idx[head]).expand(len(self.lookuptable_E), 1)
                    relation_index = torch.tensor(relation_index).expand(len(self.lookuptable_E), 1)
                    total_index = torch.unsqueeze(torch.arange(len(self.lookuptable_E)), dim=1)
                    data_index = torch.cat([head_index, relation_index, total_index], dim=1)
                    distance = self.model(data_index.to(self.device))
                    index = distance.argmin(dim=0)
                    value = distance[index]
                    score_list.append(value)
                    tail_list.append(index)
                    pass
            score = torch.tensor(score_list)
            tail = torch.tensor(tail_list)
            value, index = torch.topk(score, top, largest=False)
            tb = pt.PrettyTable()
            tb.field_names = ["Query_Head", "Rank", "Candidates_Relation", "Candidates_Tail", "Distance"]
            for i in range(len(value)):
                tb.add_row([head, i + 1, self.lookuptable_R.idx2word[index[i].item()],
                            self.lookuptable_E.idx2word[tail[index][i].item()], round(value[i].item(), 3)])
            print(tb)
