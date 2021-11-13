import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # CogKTR root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKTR root directory to PATH
# print(sys.path)

# 基本模块
import os
import torch
import random
import datetime
import numpy as np
import argparse
from torch.utils.data import RandomSampler
# cogktr模块
from cogktr import *
from cogktr.core.log import save_logger

# 超参数
random.seed(1)                     # 随机数种子
np.random.seed(1)                  # 随机数种子
torch.manual_seed(1)               # 随机数种子
torch.cuda.manual_seed_all(1)      # 随机数种子
EPOCH = 200                        # 训练的轮数
LR = 0.001                         # 学习率
WEIGHT_DECAY = 0.0001              # 正则化系数
TRAINR_BATCH_SIZE = 20000          # 训练批量大小
EMBEDDING_DIM = 100                # 形成的embedding维数
MARGIN = 1.0                       # margin大小
SAVE_STEP = None                   # 每隔几轮保存一次模型
METRIC_STEP = 1                    # 每隔几轮验证一次
 
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
args = parser.parse_args()
device = str(args.device).strip().lower().replace('cuda:', '')
cpu = device == 'cpu'
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
elif device:  # non-cpu device requested
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
    assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability


logger = save_logger("../dataset/cogktr.log")
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device('cuda:0' if torch.cuda.is_available()==True else "cpu")
# logger.info("Currently working on device {}".format(device))

# Construct the corresponding dataset
# logger.info("Currently working on dir {}".format(os.getcwd()))

data_path = '../dataset/kr/FB15k-237/raw_data'
output_path = os.path.join(*data_path.split("/")[:-1], "experimental_output/" + str(datetime.datetime.now())).replace(
    ':', '-').replace(' ', '--')
# logger.info("The output path is {}.".format(output_path))

loader = FB15K237Loader(data_path)
train_data, valid_data, test_data = loader.load_all_data()
lookuptable_E, lookuptable_R      = loader.load_all_lut()
train_data.print_table(5)
valid_data.print_table(5)
test_data.print_table(5)
lookuptable_E.print_table(5)
lookuptable_R.print_table(5)
print("data_length:\n",len(train_data),len(valid_data),len(test_data))
print("table_length:\n",len(lookuptable_E),len(lookuptable_R))

processor = FB15K237Processor(lookuptable_E,lookuptable_R)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

model = TransA(entity_dict_len=len(lookuptable_E),
               relation_dict_len=len(lookuptable_R),
               embedding_dim=EMBEDDING_DIM,
               negative_sample_method="Random_Negative_Sampling")

loss = MarginLoss(margin=MARGIN)
# loss = RotatELoss(MARGIN)
# loss =TransALoss(margin=MARGIN,relation_dict_len=len(lookuptable_R),embedding_dim=EMBEDDING_DIM)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
metric = Link_Prediction(entity_dict_len=len(lookuptable_E))
# metric = LinkRotatePrediction(entity_dict_len=len(lookuptable_E))

trainer = Kr_Trainer(
    logger=logger,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    train_sampler=train_sampler,
    valid_sampler=valid_sampler,
    trainer_batch_size=TRAINR_BATCH_SIZE,
    model=model,
    loss=loss,
    optimizer=optimizer,
    metric=metric,
    epoch=EPOCH,
    output_path=output_path,
    device=device,
    save_step=SAVE_STEP,
    metric_step=METRIC_STEP,
    save_final_model=False,
    visualization=False,
)
trainer.train()
#
# evaluator = Kr_Evaluator(
#     test_dataset=test_dataset,
#     metric=metric,
#     model_path="..\dataset\kr\FB15k-237\experimental_output/2021-11-03--16-39-49.138287\checkpoints\TransE_Model_10epochs.pkl"
# )
# evaluator.evaluate()
