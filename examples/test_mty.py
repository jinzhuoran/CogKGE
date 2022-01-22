# add cogkge directory to sys.path
from logging import config
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # CogKTR root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKTR root directory to PATH

# 基本模块
import os
import torch
import random
import argparse
import datetime
import numpy as np
import yaml
import shutil
from torch.utils.data import RandomSampler

# cogkge模块
from cogkge import *


# init the random seeds
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


init_seed(1)
parser = argparse.ArgumentParser(description="konwledge embedding toolkit")
parser.add_argument('--config',
                    default='./config_mty.yaml',
                    help='path to the configuration file')
cmd_args = parser.parse_args()

# load the arguments from the yaml file
with open(cmd_args.config, 'r') as f:
    args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))

# configure multi-gpu environment
device = str(args.device).strip().lower().replace('cuda:', '')
cpu = device == 'cpu'
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
elif device:  # non-cpu device requested
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
    assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
device = torch.device('cuda:0' if torch.cuda.is_available() == True else "cpu")

# construct the output path and log file
output_path = cal_output_path(args.data_path, args.model_name)
if not os.path.exists(output_path):
    os.makedirs(output_path)
logger = save_logger(os.path.join(output_path, "run.log"))
logger.info("Data Path:{}".format(args.data_path))
logger.info("Output Path:{}".format(output_path))
print("Data Path:{}".format(args.data_path))
print("Output Path:{}".format(output_path))

# copy the configuration and main file to experimental_output
shutil.copy(cmd_args.config, output_path)
shutil.copy(FILE, output_path)

# load the dataset
get_class = lambda name: globals()[name]
MyLoader = get_class(args.data_loader)
loader = MyLoader(args.data_path, args.download, args.download_path)
# loader = MyLoader(args.data_path)
train_data, valid_data, test_data = loader.load_all_data()
node_vocab, relation_vocab = loader.load_all_vocabs()
lookuptable_E = loader.load_all_lut()


Processor = get_class(args.data_processor)
processor = Processor(node_vocab, relation_vocab,lookuptable_E)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)

# load sampler

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

########################################################################################################################


Model = get_class(args.model_name)
model = Model(entity_dict_len=len(node_vocab),
              relation_dict_len=len(relation_vocab),
              **args.model_args)

Loss = get_class(args.loss_name)
loss = Loss(**args.loss_args)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

Metric = get_class(args.metric_name)
metric = Metric(**args.metric_args)

# Learning Rate Scheduler:
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,  # mean rank!
    factor=0.5, min_lr=1e-9, verbose=True
)


Negative_sampler = get_class(args.negative_sampler_name)
if args.negative_sampler_name == 'AdversarialSampler':
    if 'neg_per_pos' not in args.loss_args:
        assert ValueError("Please configure the neg_per_pos in loss_args if you want to choose AdversarialSampler!")
    negative_sampler = Negative_sampler(triples=train_dataset,
                                        entity_dict_len=len(node_vocab),
                                        relation_dict_len=len(relation_vocab),
                                        neg_per_pos = args.loss_args['neg_per_pos'])
else:
    negative_sampler = Negative_sampler(triples=train_dataset,
                                        entity_dict_len=len(node_vocab),
                                        relation_dict_len=len(relation_vocab))


Trainer = get_class(args.trainer_name)
trainer = Trainer(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    train_sampler=train_sampler,
    valid_sampler=valid_sampler,
    model=model,
    loss=loss,
    optimizer=optimizer,
    negative_sampler=negative_sampler,
    device=device,
    output_path=output_path,
    lookuptable_E= node_vocab,
    lookuptable_R= relation_vocab,
    metric=metric,
    lr_scheduler=lr_scheduler,
    logger=logger,
    **args.trainer_args
)
trainer.train()


Evaluator = get_class(args.evaluator_name)
evaluator = Evaluator(
    test_dataset=test_dataset,
    test_sampler=test_sampler,
    model=model,
    device=device,
    metric=metric,
    output_path=output_path,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    lookuptable_E= node_vocab,
    lookuptable_R= relation_vocab,
    logger=logger,
    **args.evaluator_args
)
evaluator.evaluate()
