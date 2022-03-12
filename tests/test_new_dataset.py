# import torch
# from torch.utils.data import RandomSampler
# from pathlib import Path
# import sys
#
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0].parents[0].parents[0]  # CogKGE root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add CogKGE root directory to PATH
# from cogkge import *
#
# loader = FB15KLoader(dataset_path="../dataset", download=False)
# train_data, valid_data, test_data = loader.load_all_data()
# node_lut, relation_lut = loader.load_all_lut()
#
# processor = FB15KProcessor(node_lut, relation_lut, reprocess=True,train_pattern="classification_based")
# # processor = FB15KProcessor(node_lut, relation_lut, reprocess=True,train_pattern="score_based")
# train_dataset = processor.process(train_data)
# valid_dataset = processor.process(valid_data)
# test_dataset = processor.process(test_data)
# node_lut, relation_lut= processor.process_lut()
#
#
# for i in range(2):
#     print(train_dataset[i])
#     print(valid_dataset[i])
#     print(test_dataset[i])
# print("Train:{}  Valid:{} Test:{}".format(len(train_dataset),
#                                           len(valid_dataset),
#                                           len(test_dataset)))

import torch
from torch.utils.data import RandomSampler
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0]  # CogKGE root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKGE root directory to PATH
from cogkge import *

loader =EVENTKG240KLoader(dataset_path="../dataset",download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut ,time_lut= loader.load_all_lut()


processor = EVENTKG240KProcessor(node_lut, relation_lut,time_lut,
                               reprocess=True,
                               nodetype=True,time=True,relationtype=True,description=True,graph=False,
                               time_unit="year",pretrain_model_name="roberta-base",token_len=10)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut,relation_lut,time_lut=processor.process_lut()


for i in range(2):
    print(train_dataset[i])
    print(valid_dataset[i])
    print(test_dataset[i])
print("Train:{}  Valid:{} Test:{}".format(len(train_dataset),
                                          len(valid_dataset),
                                          len(test_dataset)))

