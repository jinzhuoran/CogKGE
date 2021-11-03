import torch
from .log import *


class Kr_Evaluator:
    def __init__(self,
                 test_dataset,
                 metric,
                 model_path
                 ):
        self.test_dataset = test_dataset
        self.metric = metric
        self.model_path = model_path
        self.model = None

    def evaluate(self):
        self.model = torch.load(self.model_path)
        if self.metric.name == "Link_Prediction":
            self.metric(self.model, self.test_dataset)
            raw_meanrank = self.metric.raw_meanrank
            raw_hitatten = self.metric.raw_hitatten
            print("raw_meanrank:%.2f" % raw_meanrank, "raw_hitatten:%.2f%%" % raw_hitatten)
            logger.info(123)
        pass
