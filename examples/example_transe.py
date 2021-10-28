#导入基本模块
import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

#导入cogktr模块
from cogktr import *

#设置超参数
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
SAVE_STEP=10                 #每隔几轮保存一次模型
METRIC_STEP=1                #每隔几轮验证一次
METRIC_TEST_EPOCH=9          #评价重复轮数
METRIC_SAMPLE_NUM=10         #评价时采样的个数

#指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
print(torch.cuda.is_available())
cuda = torch.device('cuda:0')

#加载原始数据集
loader = FB15K237Loader("../dataset/Fb15k-237")
train_data,valid_data,test_data=loader.load_all_data()
entity2idx,relation2idx=loader.load_all_dict()

#数据处理
processor=FB15K237Processor("../dataset/Fb15k-237")
train_datable=processor.process(train_data)
valid_datable=processor.process(valid_data)
test_datable=processor.process(test_data)

#DataTableSet()暂时用如下替代
train_dataset=Data.DataLoader(dataset=train_datable,batch_size=BATCH_SIZE_TRAIN,shuffle=True)
valid_dataset=Data.DataLoader(dataset=valid_datable,batch_size=BATCH_SIZE_TRAIN,shuffle=True)
test_dataset=Data.DataLoader(dataset=test_datable,batch_size=BATCH_SIZE_TEST,shuffle=True)

#RandomSampler()还未写

model=TransE(entity_dict_len=len(entity2idx),relation_dict_len=len(relation2idx),embedding_dim=EMBEDDING_DIM,margin=MARGIN,L=L)
metric=MeanRank_HitAtTen(sample_num=METRIC_SAMPLE_NUM,test_epoch=METRIC_TEST_EPOCH,entity2idx_len=len(entity2idx))
loss =MarginLoss(entity_dict_len=len(entity2idx))
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
