#导入基本模块
import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#导入cogktr模块
from cogktr import *

#设置超参数
random.seed(1)               #随机数种子
np.random.seed(1)            #随机数种子
DATASET_FILE_PATH="FB15k-237"#文件路径
EMBEDDING_DIM=100            #形成的embedding维数
MARGIN=1.0                   #margin大小
L=2                          #范数类型
LR=0.001                     #学习率
EPOCH=10                     #训练的轮数
BATCH_SIZE=2048              #批量大小
WEIGHT_DECAY=0.0001          #正则化系数
MODEL="transe"               #选择模型
METRIC_SAMPLE_NUM=100        #评价采样个数
METRIC_TEST_EPOCH=1000       #评价重复轮数

#指定GPU

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
train_dataset=Data.DataLoader(dataset=train_datable,batch_size=BATCH_SIZE,shuffle=True)
valid_dataset=Data.DataLoader(dataset=valid_datable,batch_size=BATCH_SIZE,shuffle=True)
test_dataset=Data.DataLoader(dataset=test_datable,batch_size=BATCH_SIZE,shuffle=True)

#RandomSampler()还未写

model=TransE(entity_dict_len=len(entity2idx),relation_dict_len=len(relation2idx),embedding_dim=EMBEDDING_DIM,margin=MARGIN,L=L)
metric_1 =MeanRank(sample_num=METRIC_SAMPLE_NUM,test_epoch=METRIC_TEST_EPOCH)
metric_2 =HitTen(sample_num=METRIC_SAMPLE_NUM,test_epoch=METRIC_TEST_EPOCH)
loss =MarginLoss(entity_dict_len=len(entity2idx))
optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=WEIGHT_DECAY)

#训练部分
trainer = Trainer(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    model=model,
    metric=metric_1,
    loss=loss,
    optimizer=optimizer,
    epoch=EPOCH
)
trainer.train()
trainer.show()
# evaluator=Evaluator()
# evaluator.evaluate()
