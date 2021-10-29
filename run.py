# For testing on linux only,please don't move it.
# It contains the sample content as ./examples/example_transe.py 
# I just move that file to this directory to fix the module import error.
# user:Hongbang Yuan

#导入基本模块
import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime

import os

#导入cogktr模块
from cogktr import *

# # print all the module that are imported
# for mod in dir():
#     if not mod.endswith("__"):
#         print(mod)

#设置超参数#
random.seed(1)               #随机数种子
np.random.seed(1)            #随机数种子
EMBEDDING_DIM=100            #形成的embedding维数
MARGIN=1.0                   #margin大小
L=2                          #范数类型
LR=0.001                     #学习率
EPOCH=10                     #训练的轮数
BATCH_SIZE_TRAIN=2048        #训练批量大小
BATCH_SIZE_TEST=100          #测试批量大小
WEIGHT_DECAY=0.0001          #正则化系数
SAVE_STEP=None               #每隔几轮保存一次模型
METRIC_STEP=2                #每隔几轮验证一次
METRIC_TEST_EPOCH=10         #评价重复轮数
METRIC_SAMPLE_NUM=100        #评价时采样的个数

#指定GPU

# Construct the corresponding dataset
print("Currently working on dir ",os.getcwd())

data_path = './dataset/kr/FB15k-237/raw_data'
output_path = os.path.join(*data_path.split("/")[:-1],"experimental_output/"+str(datetime.datetime.now()))
print("the output path is {}.".format(output_path))


loader = FB15K237Loader(data_path)
lookUpTable = loader.createLUT()
processor = FB15K237Processor(lookUpTable)

train_data,valid_data,test_data = loader.load_all_data()
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
 
 
model=TransE(entity_dict_len=lookUpTable.num_entity(),relation_dict_len=lookUpTable.num_relation(),embedding_dim=EMBEDDING_DIM,margin=MARGIN,L=L)
metric=MeanRank_HitAtTen(sample_num=METRIC_SAMPLE_NUM,test_epoch=METRIC_TEST_EPOCH,entity2idx_len=lookUpTable.num_entity())
loss =MarginLoss(entity_dict_len=lookUpTable.num_entity())
optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=WEIGHT_DECAY)

#训练部分
trainer = Trainer(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    model=model,
    metric=metric,
    loss=loss,
    optimizer=optimizer,
    epoch=EPOCH,
    save_step=SAVE_STEP,
    metric_step=METRIC_STEP
)
trainer.train()
trainer.show()

#测试部分
evaluator=Evaluator(
    test_dataset=test_dataset,
    model_path="TransE_Model_10epochs.pkl",
    metric=metric
)
evaluator.evaluate()
