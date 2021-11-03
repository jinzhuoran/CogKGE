# 导入基本模块
import torch
import datetime
import os
from torch.utils.data import RandomSampler

# 导入cogktr模块
from cogktr import *

# 设置超参数#
# random.seed(1)                   # 随机数种子
# np.random.seed(1)                # 随机数种子
TRAINR_BATCH_SIZE = 20000          # 训练批量大小
EMBEDDING_DIM = 100                # 形成的embedding维数
MARGIN = 1.0                       # margin大小
EPOCH = 10                         # 训练的轮数
LR = 0.001                         # 学习率
WEIGHT_DECAY = 0.0001              # 正则化系数
SAVE_STEP = None                   # 每隔几轮保存一次模型
METRIC_STEP = 2                    # 每隔几轮验证一次

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定可用的GPU序号，将这个序列重新编号，编为0，1，2，3，后面调用的都是编号
print(torch.cuda.is_available())  # 查看cuda是否能运行
cuda = torch.device('cuda:0')  # 指定GPU序号

# Construct the corresponding dataset
print("Currently working on dir ", os.getcwd())

data_path = '../dataset/kr/FB15k-237/raw_data'
output_path = os.path.join(*data_path.split("/")[:-1], "experimental_output/" + str(datetime.datetime.now())).replace(
    ':', '-').replace(' ', '。')
print("the output path is {}.".format(output_path))

loader = FB15K237Loader(data_path)
train_data, valid_data, test_data = loader.load_all_data()
lookUpTable = loader.createLUT()

processor = FB15K237Processor(lookUpTable)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

model = TransE(entity_dict_len=lookUpTable.num_entity(),
               relation_dict_len=lookUpTable.num_relation(),
               embedding_dim=EMBEDDING_DIM,
               negative_sample_method="Random_Negative_Sampling")
loss = MarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
metric = Link_Prediction(entity_dict_len=lookUpTable.num_entity())

trainer = Kr_Trainer(
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
    save_step=SAVE_STEP,
    metric_step=METRIC_STEP,
    save_final_model=True,
    visualization=False
)
# trainer.train()

evaluator = Kr_Evaluator(
    test_dataset=test_dataset,
    metric=metric,
    model_path="..\dataset\kr\FB15k-237\experimental_output/2021-11-03。12-57-04.410190\checkpoints\TransE_Model_10epochs.pkl"
)
# evaluator.evaluate()
