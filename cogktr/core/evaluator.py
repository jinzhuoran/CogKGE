# import torch
# from .log import *
#
#
# class Evaluator:
#     def __init__(self,
#                  test_dataset,
#                  model_path,
#                  metric,
#                  ):
#         self.test_dataset = test_dataset
#         self.model_path = model_path
#         self.metric = metric
#         self.result_rank_numpy=None
#         self.mean_rank=None
#         self.hit_at_ten=None
#
#     def evaluate(self):
#         print("The evaluating process is beginning!")
#         model = torch.load(self.model_path)
#         logger.info("The model named \"%s\" has been loaded!" % (self.model_path))
#         print("Model structure:\n", model)
#         self.metric(self.test_dataset, model)
#         print("result_rank_numpy:\n", self.metric.result_rank_numpy)
#         self.result_rank_numpy=self.metric.result_rank_numpy
#         self.mean_rank=self.metric.mean_rank
#         self.hit_at_ten=self.metric.hit_at_ten / self.metric.test_epoch * 100
#         logger.info("mean_rank(total_sample_num_is_%d):%f" % (self.metric.sample_num, self.metric.mean_rank))
#         logger.info("hit_at_ten(total_sample_num_is_%d,total_epoch_num_is_%d):%f%%" % (self.metric.sample_num,self.metric.test_epoch,
#                                                                                        self.metric.hit_at_ten / self.metric.test_epoch * 100))
#         print("The evaluating process is finished!")
#         return 0
########################################################################################################################
import torch
class Kr_Evaluator:
    def __init__(self,
                 test_dataset,
                 metric,
                 model_path
                  ):
        self.test_dataset = test_dataset
        self.metric=metric
        self.model_path=model_path
        self.model=None

    def evaluate(self):
        self.model=torch.load(self.model_path)
        if self.metric.name=="Link_Prediction":
            self.metric(self.model,self.test_dataset)
            raw_meanrank=self.metric.raw_meanrank
            raw_hitatten=self.metric.raw_hitatten
            print("raw_meanrank:%.2f"%raw_meanrank,"raw_hitatten:%.2f%%"%raw_hitatten)
        pass


