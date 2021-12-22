import re
import os
import torch
import prettytable as pt
import torch.nn.functional as F
import torch.utils.data as Data
from cogktr.core import DataLoaderX



class Kr_Evaluator(object):
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
                 logger=None,
                 dataloaderX=False,
                 num_workers=None,
                 pin_memory=False,
                 ):
        self.test_dataset=test_dataset
        self.test_sampler=test_sampler
        self.evaluator_batch_size=evaluator_batch_size
        self.model=model
        self.trained_model_path=trained_model_path
        self.metric=metric
        self.device=device
        self.output_path=output_path
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.lookuptable_E=lookuptable_E
        self.lookuptable_R=lookuptable_R
        self.logger=logger
        self.dataloaderX=dataloaderX
        self.num_workers=num_workers
        self.pin_memory=pin_memory

        #Load Data
        if self.dataloaderX:
            self.test_loader = DataLoaderX(dataset=self.test_dataset, sampler=self.test_sampler,
                                           batch_size=self.evaluator_batch_size, num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)
        else:
            self.test_loader = Data.DataLoader(dataset=self.test_dataset, sampler=self.test_sampler,
                                               batch_size=self.evaluator_batch_size, num_workers=self.num_workers,
                                               pin_memory=self.pin_memory)

        #Load Lookuptable
        # TODO: add lut_loader
        #for example
        # if self.lookuptable_E and self.lookuptable_E:
        #     self.model.load_lookuotable(self.lookuptable_E, self.lookuptable_R)

        #Load Trained Model
        self.trained_epoch=0
        if self.trained_model_path:
            if os.path.exists(self.trained_model_path):
                string=self.trained_model_path
                pattern=r"^.*?/checkpoints/.*?_(.*?)epochs$"
                match = re.search(pattern, string)
                self.trained_epoch=int(match.group(1))
                self.model.load_state_dict(torch.load(os.path.join(self.trained_model_path,"Model.pkl")))
            else:
                raise FileExistsError("Trained_model_path doesn't exist!")

        #Load Dataparallel Training
        print("Available cuda devices:", torch.cuda.device_count())
        self.parallel_model = torch.nn.DataParallel(self.model)
        self.parallel_model = self.parallel_model.to(self.device)

    def evaluate(self):
        current_epoch=self.trained_epoch

        print("Evaluating Model {} on Test Dataset...".format(self.model.name))
        self.metric.caculate(device=self.device,
                             model=self.parallel_model,
                             total_epoch=self.trained_epoch,
                             current_epoch=current_epoch,
                             metric_type="test_dataset",
                             metric_dataset=self.test_dataset,
                             node_dict_len=len(self.lookuptable_E),
                             model_name=self.model.name,
                             logger=self.logger,
                             train_dataset=self.train_dataset,
                             valid_dataset=self.valid_dataset,
                             test_dataset=self.test_dataset)
        self.metric.print_current_table()
        self.metric.log()
        self.metric.write()
    def search_similar_entity(self,entity,top):
        print("Search_similar_entity Process...")
        entity_index=self.lookuptable_E.word2idx[entity]
        entity_embedding=torch.unsqueeze(self.model.entity_embedding(torch.tensor(entity_index).to(self.device)),dim=0)
        total_index=torch.arange(len(self.lookuptable_E))
        total_embedding=self.model.entity_embedding(total_index.to(self.device))
        distance=F.pairwise_distance(entity_embedding, total_embedding, p=2)
        value,index=torch.topk(distance,top+1,largest=False)
        tb = pt.PrettyTable()
        tb.field_names = ["Query_Entity","Rank","Candidates","Distance"]
        for i in range(len(value)-1):
            tb.add_row([entity,i+1,self.lookuptable_E.idx2word[index[i+1].item()],round(value[i+1].item(),3)])
        print(tb)
    def search_similar_head(self,tail,relation,top):
        print("Search_similar_head Process...")
        tail_index=torch.tensor(self.lookuptable_E.word2idx[tail]).expand(len(self.lookuptable_E),1)
        relation_index=torch.tensor(self.lookuptable_R.word2idx[relation]).expand(len(self.lookuptable_E),1)
        total_index=torch.unsqueeze(torch.arange(len(self.lookuptable_E)),dim=1)
        data_index=torch.cat([total_index,relation_index,tail_index],dim=1)
        distance=self.model(data_index.to(self.device))
        value,index=torch.topk(distance,top,largest=False)
        tb = pt.PrettyTable()
        tb.field_names = ["Query_Relation","Query_Tail","Rank","Candidates_Head","Distance"]
        for i in range(len(value)):
            tb.add_row([relation,tail,i+1,self.lookuptable_E.idx2word[index[i].item()],round(value[i].item(),3)])
        print(tb)
    def search_similar_tail(self,head,relation,top):
        print("Search_similar_tail Process...")
        head_index=torch.tensor(self.lookuptable_E.word2idx[head]).expand(len(self.lookuptable_E),1)
        relation_index=torch.tensor(self.lookuptable_R.word2idx[relation]).expand(len(self.lookuptable_E),1)
        total_index=torch.unsqueeze(torch.arange(len(self.lookuptable_E)),dim=1)
        data_index=torch.cat([head_index,relation_index,total_index],dim=1)
        distance=self.model(data_index.to(self.device))
        value,index=torch.topk(distance,top,largest=False)
        tb = pt.PrettyTable()
        tb.field_names = ["Query_Head","Query_Relation","Rank","Candidates_Tail","Distance"]
        for i in range(len(value)):
            tb.add_row([head,relation,i+1,self.lookuptable_E.idx2word[index[i].item()],round(value[i].item(),3)])
        print(tb)


