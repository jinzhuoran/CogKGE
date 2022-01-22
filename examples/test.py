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

def parse_configs():
    """Parse Command Line arguments

    Returns:
        argparse.Namespace: Command Line Namespace
    """
    parser = argparse.ArgumentParser(description="Konwledge Embedding Toolkit")
    parser.add_argument('--config',
                    default='./config.yaml',
                    help='path to the configuration file')
    return parser.parse_args()


class ReadArgs:
    def __init__(self,cmd_args):
        """
        Initilize according to the configuration file specified by the cmd argument.

        Args:
            cmd_args (argparse.Namespace): the command line args from the parse_config() function
        """
        self.config_file = cmd_args.config
        with open(self.config_file,'r') as f:
            args = argparse.Namespace(**yaml.load(f,Loader=yaml.FullLoader))
        self.args = args
        self.get_class = lambda name: globals()[name]
        
        self.device = None
        self.output_path = None
        self.logger = None
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None
        self.lookuptable_E = None
        self.lookuptable_R = None
        
        self.model = None
        self.loss = None
        self.metric = None
        self.optimizer = None
        self.lr_scheduler = None
        self.negative_sampelr = None
        
        self.trainer = None
    
    def setDevice(self):
        """
        configure the multi-gpu environment
        """
        device = str(self.args.device).strip().lower().replace('cuda:','')
        cpu = device == 'cpu'
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
        device = torch.device('cuda:0' if torch.cuda.is_available() == True else "cpu")
        self.device = device
        
    def setLog(self):
        """
        Initialize logger and experimental output folder
        """
        output_path = cal_output_path(self.args.data_path, self.args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logger = save_logger(os.path.join(output_path, "run.log"))
        logger.info("Data Path:{}".format(self.args.data_path))
        logger.info("Output Path:{}".format(output_path))
        
        # copy the configuration and main file to experimental_output
        shutil.copy(self.config_file, output_path)
        shutil.copy(FILE, output_path)
        self.logger = logger
        self.output_path = output_path
    
    def loadDataset(self):
        """
        load the dataset from the specified datapath

        Returns:
            [type]: [description]
        """
        # load the dataset
        MyLoader = self.get_class(self.args.data_loader)
        loader = MyLoader(self.args.data_path, self.args.download, self.args.download_path)
        train_data, valid_data, test_data = loader.load_all_data()
        # lookuptable_E, lookuptable_R = loader.load_all_lut()
        # train_data.print_table(5)
        # valid_data.print_table(5)
        # test_data.print_table(5)
        # lookuptable_E.print_table(5)
        # lookuptable_R.print_table(5)
        print("data_length:\n", len(train_data), len(valid_data), len(test_data))
        # print("table_length:\n", len(lookuptable_E), len(lookuptable_R))

        Processor = self.get_class(self.args.data_processor)
        # processor = Processor(lookuptable_E, lookuptable_R)
        processor = Processor()
        train_dataset = processor.process(train_data)
        valid_dataset = processor.process(valid_data)
        test_dataset = processor.process(test_data)
        
        self.train_sampler = RandomSampler(train_dataset)
        self.valid_sampler = RandomSampler(valid_dataset)
        self.test_sampler = RandomSampler(test_dataset)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        # self.lookuptable_E,self.lookuptable_R = lookuptable_E,lookuptable_R
    
    def loadModel(self):
        Model = self.get_class(self.args.model_name)
        self.model = Model(entity_dict_len=len(self.lookuptable_E),
                           relation_dict_len=len(self.lookuptable_R),
                           **self.args.model_args)
        # return self.model
    
    def loadLoss(self):
        Loss = self.get_class(self.args.loss_name)
        self.loss = Loss(**self.args.loss_args)

    def loadMetric(self):
        Metric = self.get_class(self.args.metric_name)
        self.metric = Metric(entity_dict_len=len(self.lookuptable_E),**self.args.metric_args)

    def loadOptimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def loadLRScheduler(self):
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,  # mean rank!
            factor=0.5, min_lr=1e-9, verbose=True
            )
        self.lr_scheduler = lr_scheduler
        
    def loadNegativeSampler(self):
        Negative_sampler = self.get_class(self.args.negative_sampler_name)
        if self.args.negative_sampler_name == 'AdversarialSampler':
            if 'neg_per_pos' not in self.args.loss_args:
                assert ValueError("Please configure the neg_per_pos in loss_args if you want to choose "
                                  "AdversarialSampler!")
            negative_sampler = Negative_sampler(triples=self.train_dataset.data_numpy,
                                                entity_dict_len=len(self.lookuptable_E),
                                                relation_dict_len=len(self.lookuptable_R),
                                                neg_per_pos = self.args.loss_args['neg_per_pos'],)
        else:
            negative_sampler = Negative_sampler(triples=self.train_dataset.data_numpy,
                                            entity_dict_len=len(self.lookuptable_E),
                                            relation_dict_len=len(self.lookuptable_R))
         
        self.negative_sampler = negative_sampler
        
    def loadTrainer(self):
        Trainer = self.get_class(self.args.trainer_name)
        trainer = Trainer(
            logger=self.logger,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            test_dataset=self.test_dataset,
            train_sampler=self.train_sampler,
            valid_sampler=self.valid_sampler,
            test_sampler=self.test_sampler,
            negative_sampler=self.negative_sampler,
            model=self.model,
            loss=self.loss,
            optimizer=self.optimizer,
            metric=self.metric,
            output_path=self.output_path,
            device=self.device,
            lr_scheduler=self.lr_scheduler,
            dataloaderX=True,
            num_workers=4,
            pin_memory=True,
            **self.args.trainer_args
        )
        self.trainer = trainer
    
    def getTrainer(self):
        return self.trainer
    
        
    
def main():
    init_seed(1)
    cmd_args = parse_configs()
    readargs = ReadArgs(cmd_args)
    readargs.setDevice()
    readargs.setLog()
    readargs.loadDataset()
    readargs.loadModel()
    readargs.loadLoss()
    readargs.loadMetric()
    readargs.loadOptimizer()
    readargs.loadLRScheduler()
    readargs.loadNegativeSampler()
    readargs.loadTrainer()
    trainer = readargs.getTrainer()
    trainer.train()

    

if __name__ == '__main__':
    main()
    
    
# init_seed(1)
# parser = 
# parser.add_argument('--config',
#                     default='./config.yaml',
#                     help='path to the configuration file')
# cmd_args = parser.parse_args()

# # load the arguments from the yaml file
# with open(cmd_args.config, 'r') as f:
#     args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))

# # configure multi-gpu environment
# device = str(args.device).strip().lower().replace('cuda:', '')
# cpu = device == 'cpu'
# if cpu:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
# elif device:  # non-cpu device requested
#     os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
#     assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
# device = torch.device('cuda:0' if torch.cuda.is_available() == True else "cpu")

# # construct the output path and log file
# output_path = cal_output_path(args.data_path, args.model_name)
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# logger = save_logger(os.path.join(output_path, "run.log"))
# logger.info("Data Path:{}".format(args.data_path))
# logger.info("Output Path:{}".format(output_path))
# print("Data Path:{}".format(args.data_path))
# print("Output Path:{}".format(output_path))

# # copy the configuration and main file to experimental_output
# shutil.copy(cmd_args.config, output_path)
# shutil.copy(FILE, output_path)

# # load the dataset
# get_class = lambda name: globals()[name]
# MyLoader = get_class(args.data_loader)
# loader = MyLoader(args.data_path, args.download, args.download_path)
# # loader = MyLoader(args.data_path)
# train_data, valid_data, test_data = loader.load_all_data()
# lookuptable_E, lookuptable_R = loader.load_all_lut()
# train_data.print_table(5)
# valid_data.print_table(5)
# test_data.print_table(5)
# lookuptable_E.print_table(5)
# lookuptable_R.print_table(5)
# print("data_length:\n", len(train_data), len(valid_data), len(test_data))
# print("table_length:\n", len(lookuptable_E), len(lookuptable_R))

# Processor = get_class(args.data_processor)
# processor = Processor(lookuptable_E, lookuptable_R)
# train_dataset = processor.process(train_data)
# valid_dataset = processor.process(valid_data)
# test_dataset = processor.process(test_data)

# # load sampler

# train_sampler = RandomSampler(train_dataset)
# valid_sampler = RandomSampler(valid_dataset)
# test_sampler = RandomSampler(test_dataset)

# Model = get_class(args.model_name)
# model = Model(entity_dict_len=len(lookuptable_E),
#               relation_dict_len=len(lookuptable_R),
#               **args.model_args)

# Loss = get_class(args.loss_name)
# loss = Loss(**args.loss_args)

# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Metric = get_class(args.metric_name)
# metric = Metric(entity_dict_len=len(lookuptable_E),**args.metric_args)
# # metric = Metric(entity_dict_len=len(lookuptable_E))

# # Learning Rate Scheduler:
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,  # mean rank!
#     factor=0.5, min_lr=1e-9, verbose=True
# )
# # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
# #     optimizer,milestones=[30,60,90],gamma=0.5
# # )

# Negative_sampler = get_class(args.negative_sampler_name)
# if args.negative_sampler_name == 'AdversarialSampler':
#     if 'neg_per_pos' not in args.loss_args:
#         assert ValueError("Please configure the neg_per_pos in loss_args if you want to choose AdversarialSampler!")
#     negative_sampler = Negative_sampler(triples=train_dataset.data_numpy,
#                                         entity_dict_len=len(lookuptable_E),
#                                         relation_dict_len=len(lookuptable_R),
#                                         neg_per_pos = args.loss_args['neg_per_pos'],)
# else:
#     negative_sampler = Negative_sampler(triples=train_dataset.data_numpy,
#                                         entity_dict_len=len(lookuptable_E),
#                                         relation_dict_len=len(lookuptable_R))


