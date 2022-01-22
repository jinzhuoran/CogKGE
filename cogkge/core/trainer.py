import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
from openTSNE import TSNE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from accelerate import Accelerator
from cogkge.core import DataLoaderX
from .log import save_logger
from ..utils.kr_utils import cal_output_path


# from .log import *


class Trainer(object):
    def __init__(self,
                 train_dataset,
                 train_sampler,
                 trainer_batch_size,
                 model,
                 loss,
                 optimizer,
                 negative_sampler,
                 epoch,
                 device,
                 output_path,
                 valid_dataset=None,
                 valid_sampler=None,
                 lookuptable_E=None,
                 lookuptable_R=None,
                 metric=None,
                 lr_scheduler=None,
                 log=None,
                 load_checkpoint=None,
                 visualization=False,
                 apex=False,
                 dataloaderX=False,
                 num_workers=None,
                 pin_memory=False,
                 metric_step=None,
                 save_step=None,
                 metric_final_model=True,
                 save_final_model=True,
                 ):
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.trainer_batch_size = trainer_batch_size
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.negative_sampler = negative_sampler
        self.epoch = epoch
        self.device = device
        self.valid_dataset = valid_dataset
        self.valid_sampler = valid_sampler
        self.lookuptable_E = lookuptable_E
        self.lookuptable_R = lookuptable_R
        self.metric = metric
        self.lr_scheduler = lr_scheduler
        self.log = log
        self.load_checkpoint = load_checkpoint
        self.visualization = visualization
        self.apex = apex
        self.dataloaderX = dataloaderX
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.metric_step = metric_step
        self.save_step = save_step
        self.metric_final_model = metric_final_model
        self.save_final_model = save_final_model

        self.data_name = train_dataset.data_name

        self.visual_num = 100  # 降维可视化的样本个数
        if self.lookuptable_E.type is not None:
            self.visual_type = self.lookuptable_E.type[:self.visual_num].numpy()
        else:
            self.visual_type = np.arctan2(np.random.normal(0, 2, self.visual_num),
                                          np.random.normal(0, 2, self.visual_num))

        # Set output_path
        output_path = os.path.join(output_path, "kr", self.data_name)
        self.output_path = cal_output_path(output_path, self.model.name)
        self.output_path = self.output_path + "--{}epochs".format(self.epoch)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Set logger
        if log:
            logger = save_logger(os.path.join(self.output_path, "trainer_run.log"))
            logger.info("Data Experiment Output Path:{}".format(self.output_path))
            self.logger = logger

        # Load Apex
        if self.apex:
            if "apex" not in sys.modules:
                logger.info("Apex has not been installed!Force the parameter to be False.")
                self.apex = False
            else:
                from apex import amp
                self.model, self.optimizer = amp.initialize(self.model.to(self.device), self.optimizer, opt_level="O1")

        # if self.apex:
        #     try:
        #         from apex import amp
        #     except ImportError:
        #         raise ImportError("Please install apex.")
        #     self.model , self.optimizer = amp.initialize(self.model.to(self.device) , self.optimizer, opt_level="O1")

        # Load Data
        if self.dataloaderX:
            self.train_loader = DataLoaderX(dataset=self.train_dataset, sampler=self.train_sampler,
                                            batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                            pin_memory=self.pin_memory)
            if self.valid_dataset:
                self.valid_loader = DataLoaderX(dataset=self.valid_dataset, sampler=self.valid_sampler,
                                                batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                                pin_memory=self.pin_memory)
        else:
            self.train_loader = Data.DataLoader(dataset=self.train_dataset, sampler=self.train_sampler,
                                                batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                                pin_memory=self.pin_memory)
            if self.valid_dataset:
                self.valid_loader = Data.DataLoader(dataset=self.valid_dataset, sampler=self.valid_sampler,
                                                    batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                                    pin_memory=self.pin_memory)

        # Load Lookuptable
        # TODO: add lut_loader
        # for example
        # if self.lookuptable_E and self.lookuptable_E:
        #     self.model.load_lookuotable(self.lookuptable_E, self.lookuptable_R)

        # Load Checkpoint
        self.trained_epoch = 0
        if self.load_checkpoint:
            if os.path.exists(self.load_checkpoint):
                string = self.load_checkpoint
                pattern = r"^.*?/checkpoints/.*?_(.*?)epochs$"
                match = re.search(pattern, string)
                self.trained_epoch = int(match.group(1))
                self.model.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Model.pkl")))
                self.optimizer.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Optimizer.pkl")))
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Lr_Scheduler.pkl")))
            else:
                raise FileExistsError("Checkpoint path doesn't exist!")

        # Load Dataparallel Training
        # self.accelerator = Accelerator()
        # self.device = self.accelerator.device
        # self.model, my_optimizer, my_training_dataloader = self.accelerate.prepare(
        #     +     my_model, my_optimizer, my_training_dataloader)
        print("Available cuda devices:", torch.cuda.device_count())
        self.parallel_model = torch.nn.DataParallel(self.model)
        self.parallel_model = self.parallel_model.to(self.device)

        # Load Visualization
        self.writer = None
        if self.visualization == True:
            self.visualization_path = os.path.join(self.output_path, "visualization", self.model.name)
            if not os.path.exists(self.visualization_path):
                os.makedirs(self.visualization_path)
            self.writer = SummaryWriter(self.visualization_path)
            self.logger.info(
                "The visualization path is" + self.visualization_path)

        # Load Metric
        if self.metric:
            self.metric.initialize(device=self.device,
                                   total_epoch=self.epoch,
                                   metric_type="valid",
                                   node_dict_len=len(self.lookuptable_E),
                                   model_name=self.model.name,
                                   logger=self.logger,
                                   writer=self.writer,
                                   train_dataset=self.train_dataset,
                                   valid_dataset=self.valid_dataset)
            if self.metric.link_prediction_filt:
                self.metric.establish_correct_triplets_dict()

    def train(self):
        if self.epoch - self.trained_epoch <= 0:
            raise ValueError("Trained_epoch is bigger than total_epoch!")

        for epoch in range(self.epoch - self.trained_epoch):
            current_epoch = epoch + 1 + self.trained_epoch

            # Training Progress
            train_epoch_loss = 0.0
            for train_step, train_positive in enumerate(tqdm(self.train_loader)):
                train_positive = train_positive.to(self.device)
                train_negative = self.negative_sampler.create_negative(train_positive[:, :3])
                train_positive_score = self.parallel_model(train_positive)
                if len(train_positive[0]) == 5:
                    train_negative = torch.cat((train_negative, train_positive[:, 3:]), dim=1)
                train_negative_score = self.parallel_model(train_negative)
                penalty = self.model.get_penalty() if hasattr(self.model, 'get_penalty') else 0
                train_loss = self.loss(train_positive_score, train_negative_score, penalty)
                # self.accelerate.backward(train_loss)

                self.optimizer.zero_grad()
                if self.apex:
                    from apex import amp
                    with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    train_loss.backward()
                train_epoch_loss = train_epoch_loss + train_loss.item()
                self.optimizer.step()

            valid_epoch_loss = 0.0
            with torch.no_grad():
                for valid_step, valid_positive in enumerate(self.valid_loader):
                    valid_positive = valid_positive.to(self.device)
                    valid_negative = self.negative_sampler.create_negative(valid_positive[:, :3])
                    valid_positive_score = self.parallel_model(valid_positive)
                    if len(valid_positive[0]) == 5:
                        valid_negative = torch.cat((valid_negative, valid_positive[:, 3:]), dim=1)
                    valid_negative_score = self.parallel_model(valid_negative)
                    penalty = self.model.get_penalty() if hasattr(self.model, 'get_penalty') else 0
                    valid_loss = self.loss(valid_positive_score, valid_negative_score, penalty)
                    valid_epoch_loss = valid_epoch_loss + valid_loss.item()

            print("Epoch{}/{}   Train Loss:".format(current_epoch, self.epoch), train_epoch_loss / (train_step + 1),
                  " Valid Loss:", valid_epoch_loss / (valid_step + 1))

            # Metric Progress
            if self.metric_step and (current_epoch) % self.metric_step == 0 or self.metric_final_model and (
                    current_epoch) == self.epoch:
                print("Evaluating Model {} on Valid Dataset...".format(self.model.name))
                self.metric.caculate(model=self.parallel_model, current_epoch=current_epoch)
                self.metric.print_current_table()
                self.metric.log()
                self.metric.write()
                print("-----------------------------------------------------------------------")

                # Scheduler Progress
                self.lr_scheduler.step(self.metric.get_Raw_MR())

            # Visualization Process
            if self.visualization:
                self.writer.add_scalars("Loss", {"train_loss": train_epoch_loss,
                                                 "valid_loss": valid_epoch_loss}, current_epoch)
                # img = np.zeros((3, 100, 100))
                # img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
                # img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
                # self.writer.add_image("Dimensionality reduction image", img, current_epoch)
                # image=np.ones((64,64,3))*random.randint(255)
                # self.writer.add_image("Dimensionality reduction image", image,current_epoch , dataformats='HWC')
                embedding = self.model.entity_embedding.weight.data.clone().cpu().numpy()[:self.visual_num]
                embedding = TSNE(negative_gradient_method="bh").fit(embedding)
                plt.scatter(embedding[:, 0], embedding[:, 1], c=self.visual_type)
                plt.show()
                if epoch == 0:
                    fake_data = torch.zeros(self.trainer_batch_size, 3).long()
                    self.writer.add_graph(self.model.cpu(), fake_data)
                    self.model.to(self.device)
                # for name, param in self.model.named_parameters():
                #     self.writer.add_histogram(name + '_grad', param.grad, epoch)
                #     self.writer.add_histogram(name + '_data', param, epoch)
                # if epoch == 0:
                #     embedding_data = torch.rand(10, 20)
                #     embedding_label = ["篮球", "足球", "乒乓球", "羽毛球", "保龄球", "游泳", "爬山", "旅游", "赛车", "写代码"]
                #     self.writer.add_embedding(mat=embedding_data, metadata=embedding_label)

            # Save Checkpoint and Final Model Process
            if (self.save_step and (current_epoch) % self.save_step == 0) or (current_epoch) == self.epoch:
                if not os.path.exists(os.path.join(self.output_path, "checkpoints",
                                                   "{}_{}epochs".format(self.model.name, current_epoch))):
                    os.makedirs(os.path.join(self.output_path, "checkpoints",
                                             "{}_{}epochs".format(self.model.name, current_epoch)))
                    self.logger.info(os.path.join(self.output_path, "checkpoints",
                                                  "{}_{}epochs ".format(self.model.name,
                                                                        current_epoch)) + 'created successfully!')
                torch.save(self.model.state_dict(), os.path.join(self.output_path, "checkpoints",
                                                                 "{}_{}epochs".format(self.model.name, current_epoch),
                                                                 "Model.pkl"))
                torch.save(self.optimizer.state_dict(), os.path.join(self.output_path, "checkpoints",
                                                                     "{}_{}epochs".format(self.model.name,
                                                                                          current_epoch),
                                                                     "Optimizer.pkl"))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(self.output_path, "checkpoints",
                                                                        "{}_{}epochs".format(self.model.name,
                                                                                             current_epoch),
                                                                        "Lr_Scheduler.pkl"))
                self.logger.info(os.path.join(self.output_path, "checkpoints", "{}_{}epochs ".format(self.model.name,
                                                                                                     current_epoch)) + "saved successfully")

        # Show Best Metric Result
        if self.metric_step:
            self.metric.print_best_table(front=5, key="Raw_MR")
