import torch
import os
import re
class Kr_Evaluator:
    def __init__(self,
                 logger,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 lookuptable_E,
                 lookuptable_R,
                 model,
                 metric,
                 device,
                 load_checkpoint,
                 ):
        self.logger=logger
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.test_dataset=test_dataset
        self.lookuptable_E=lookuptable_E
        self.lookuptable_R=lookuptable_R
        self.model=model
        self.metric=metric
        self.device=device
        self.load_checkpoint=load_checkpoint
    def evaluate(self):
        if os.path.exists(self.load_checkpoint):
            parallel_model=torch.nn.DataParallel(torch.load(self.load_checkpoint))
            parallel_model=parallel_model.to(self.device)
        else:
            raise FileExistsError("Checkpoint path doesn't exist!")

        string=self.load_checkpoint
        pattern=r"^.*?/checkpoints/(.*?)_(.*?)_(.*?)epochs.pkl$$"
        match = re.search(pattern, string)
        model_name=match.group(1)
        epoch=match.group(3)

        print("Evaluating Model {}...".format(self.model.name))
        self.metric.caculate(device=self.device,
                             model=parallel_model,
                             total_epoch=epoch,
                             current_epoch=epoch,
                             metric_type="test_dataset",
                             metric_dataset=self.test_dataset,
                             node_dict_len=len(self.lookuptable_E),
                             model_name=model_name,
                             logger=self.logger,
                             writer=None,
                             train_dataset=self.train_dataset,
                             valid_dataset=self.valid_dataset,
                             test_dataset=self.test_dataset)
        self.metric.print_table()
        self.metric.log()
        self.metric.write()
