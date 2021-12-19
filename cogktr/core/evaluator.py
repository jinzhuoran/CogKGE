import re
import os
import torch
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


