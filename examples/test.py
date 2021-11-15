# add cogktr directory to sys.path
from logging import config
import sys
from pathlib import Path
def add_path():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0].parents[0]  # CogKTR root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add CogKTR root directory to PATH
add_path()

# 基本模块
import os
import torch
import random
import argparse
import datetime
import numpy as np
import yaml
from torch.utils.data import RandomSampler

# cogktr模块
from cogktr import *

# init the random seeds
def init_seed(seed):
    random.seed(seed)                     
    np.random.seed(seed)                  
    torch.manual_seed(seed)                
    torch.cuda.manual_seed_all(seed)      

init_seed(1)
parser = argparse.ArgumentParser(description="konwledge embedding toolkit")
parser.add_argument('--config',
                    default='./config.yaml',
                    help='path to the configuration file')
cmd_args = parser.parse_args()

# load the arguments from the yaml file
with open(cmd_args.config,'r') as f:
    args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))

# configure multi-gpu environment
device = str(args.device).strip().lower().replace('cuda:', '')
cpu = device == 'cpu'
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
elif device:  # non-cpu device requested
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
    assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
device = torch.device('cuda:0' if torch.cuda.is_available()==True else "cpu")

# construct the output path and log file
output_path = cal_output_path(args.data_path,args.model_name)
if not os.path.exists(output_path):
    os.makedirs(output_path)
logger = save_logger(os.path.join(output_path,"run.log"))
logger.info("Data Path:{}".format(args.data_path))
logger.info("Output Path:{}".format(output_path))

# load the dataset
get_class = lambda name : globals()[name]
MyLoader = get_class(args.data_loader)
loader = MyLoader(args.data_path)
train_data, valid_data, test_data = loader.load_all_data()
lookuptable_E, lookuptable_R      = loader.load_all_lut()
train_data.print_table(5)
valid_data.print_table(5)
test_data.print_table(5)
lookuptable_E.print_table(5)
lookuptable_R.print_table(5)
print("data_length:\n",len(train_data),len(valid_data),len(test_data))
print("table_length:\n",len(lookuptable_E),len(lookuptable_R))

Processor = get_class(args.data_processor)
processor = Processor(lookuptable_E,lookuptable_R)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

Model = get_class(args.model_name)
model = Model(entity_dict_len=len(lookuptable_E),
              relation_dict_len=len(lookuptable_R),
              **args.model_args)

Loss = get_class(args.loss_name)
loss = Loss(**args.loss_args)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

Metric = get_class(args.metric_name)
metric = Metric(entity_dict_len=len(lookuptable_E))

Trainer = get_class(args.trainer_name)
trainer = Trainer(
    logger=logger,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    train_sampler=train_sampler,
    valid_sampler=valid_sampler,
    model=model,
    loss=loss,
    optimizer=optimizer,
    metric=metric,
    output_path=output_path,
    device=device,
    **args.trainer_args
)
trainer.train()




