import torch
from .log import *


class Evaluator:
    def __init__(self,
                 test_dataset,
                 model_path,
                 metric,
                 ):
        self.test_dataset = test_dataset
        self.model_path = model_path
        self.metric = metric
        self.mean_rank=None
        self.hit_at_ten=None

    def evaluate(self):
        print("The evaluating process is beginning!")
        model = torch.load(self.model_path)
        logger.info("The model named \"%s\" has been loaded!" % (self.model_path))
        print("Model structure:\n", model)
        self.metric(self.test_dataset, model)
        print("result_rank_numpy:\n", self.metric.result_rank_numpy)
        self.mean_rank=self.metric.mean_rank
        self.hit_at_ten=self.metric.hit_at_ten
        logger.info("mean_rank(total_sample_num_is_%d):%f" % (self.metric.sample_num, self.metric.mean_rank))
        logger.info("hit_at_ten(total_sample_num_is_%d,total_epoch_num_is_%d):%f%%" % (self.metric.sample_num,self.metric.test_epoch,
                                                                                       self.metric.hit_at_ten / self.metric.test_epoch * 100))
        print("The evaluating process is finished!")
        return 0
