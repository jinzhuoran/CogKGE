#导入模块原始模块
import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#导入cogktr
from cogktr import *

#设置超参数
random.seed(1)               #随机数种子
DATASET_FILE_PATH="FB15k-237"#文件路径
EMBEDDING_DIM=100            #形成的embedding维数
MARGIN=1.0                   #margin大小
L=2                          #范数类型
LR=0.001                      #学习率
EPOCH=50                     #训练的轮数
BATCH_SIZE=2048              #批量大小
WEIGHT_DECAY=0.0001           #正则化系数
MODEL="transe"               #选择模型

#载入原始数据集
loader=Loader()