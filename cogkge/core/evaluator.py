import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cogkge.core import DataLoaderX
from .log import save_logger
from ..utils.kr_utils import cal_output_path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def reduce_mean(tensor,nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt = rt/nprocs
    return rt

class Evaluator(object):
    def __init__(self,
                 train_dataset,
                 train_sampler,
                 valid_dataset,
                 valid_sampler,
                 model,
                 loss,
                 metric,
                 negative_sampler,
                 optimizer,
                 device,
                 test_dataset,
                 test_sampler,
                 checkpoint_path,
                 lookuptable_E=None,
                 lookuptable_R=None,
                 time_lut=None,
                 lr_scheduler=None,
                 apex=False,
                 dataloaderX=False,
                 num_workers=0,
                 pin_memory=True,
                 use_metric_epoch=0.1,
                 use_tensorboard_epoch=0.1,
                 use_savemodel_epoch=0.1,
                 use_matplotlib_epoch=0.1,
                 rank=-1,
                 ):
        # 传入参数
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.valid_dataset = valid_dataset
        self.valid_sampler = valid_sampler
        self.model = model
        self.loss = loss
        self.metric = metric
        self.negative_sampler = negative_sampler
        self.optimizer = optimizer
        self.device = device
        # self.output_path = cal_output_path(os.path.join(output_path, train_dataset.data_name),
        #                                    self.model.model_name) + "--{}epochs".format(total_epoch)
        self.test_dataset=test_dataset
        self.test_sampler = test_sampler
        self.lookuptable_E = lookuptable_E
        self.lookuptable_R = lookuptable_R
        self.time_lut=time_lut
        self.lr_scheduler=lr_scheduler
        self.apex = apex
        self.dataloaderX = dataloaderX
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_metric_epoch = use_metric_epoch
        self.use_tensorboard_epoch = use_tensorboard_epoch
        self.use_savemodel_epoch = use_savemodel_epoch
        self.use_matplotlib_epoch = use_matplotlib_epoch
        self.checkpoint_path = checkpoint_path

        # 全局变量
        self.average_train_epoch_loss_list = []  # 平均训练损失
        self.average_valid_epoch_loss_list = []  # 平均验证损失
        self.current_epoch_list = []  # 目前轮数list
        self.writer = None  # tensorboard
        self.trained_epoch = 0  # 已经训练的轮数
        self.logger = None  # log
        # self.visualization_path = os.path.join(self.output_path, "visualization", self.model.model_name) #可视化路径
        # self.log_path = os.path.join(self.output_path, "trainer_run.log") #log路径
        self.data_name = train_dataset.data_name
        self.model_name = self.model.model_name
        self.rank = rank
        self.best_metric=None #最好的验证指标
        self.best_epoch=0 #最好的轮数

        # # Set path
        # if self.rank in [-1,0]:
        #     if not os.path.exists(self.output_path):
        #         os.makedirs(self.output_path)
        #     if not os.path.exists(self.visualization_path):
        #         os.makedirs(self.visualization_path)

        # Set Model
        time_dict_len=0
        nodetype_dict_len=0
        relationtype_dict_len=0
        if hasattr(time_lut, "vocab") and time_lut.vocab is not None:
            time_dict_len = len(time_lut.vocab)
        if hasattr(lookuptable_E, "type") and lookuptable_E.type is not None:
            nodetype_dict_len = len(set(lookuptable_E.type.numpy()))
        if hasattr(lookuptable_R, "type") and lookuptable_R.type is not None:
            relationtype_dict_len = len(set(lookuptable_R.type.numpy()))

        # time_dict_len=len(time_lut.vocab) if hasattr(time_lut,"vocab") else 0
        # nodetype_dict_len = len(set(lookuptable_E.type.numpy()))if hasattr(lookuptable_E,"type") else 0
        # relationtype_dict_len = len(set(lookuptable_R.type.numpy())) if hasattr(lookuptable_R, "type") else 0
        self.model.set_model_config(model_loss=self.loss,
                                    model_metric=metric,
                                    model_negative_sampler=negative_sampler,
                                    model_device=self.device,
                                    time_dict_len=time_dict_len,
                                    nodetype_dict_len=nodetype_dict_len,
                                    relationtype_dict_len=relationtype_dict_len)
        # Set Checkpoint
        if self.checkpoint_path != None:
            if os.path.exists(self.checkpoint_path):
                # string = self.checkpoint_path
                # pattern = r"^.*?/checkpoints/.*?_(.*?)epochs$"
                # match = re.search(pattern, string)
                # self.trained_epoch = int(match.group(1))

                model_state_dict_path = os.path.abspath(os.path.join(self.checkpoint_path, "Model.pkl"))
                print("Load model state dict from {}".format(model_state_dict_path))
                self.model.load_state_dict(torch.load(model_state_dict_path))

                # optimizer_state_dict_path = os.path
                # print("Load optimizer state dict from {}".format(os.path.join(self.checkpoint_path, "Optimizer.pkl")))
                # self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "Optimizer.pkl")))
                #
                # print("Load lr_scheduler state dict from {}".format(os.path.join(self.checkpoint_path, "Lr_Scheduler.pkl")))
                # self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "Lr_Scheduler.pkl")))
            else:
                raise FileExistsError("Checkpoint path doesn't exist!")

        if self.rank == -1:
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.cuda(self.rank)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.rank],
                             output_device=self.rank,
                             find_unused_parameters=False,
                             broadcast_buffers=False)

        # Set Apex
        if self.apex:
            if "apex" not in sys.modules:
                # print("Please install apex!")
                self.apex = False
            else:
                from apex import amp
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        # Set log
        # self.logger = save_logger(self.log_path,rank=self.rank)
        # self.logger.info("Data Experiment Output Path:{}".format(os.path.abspath(self.output_path)))

        # Set Tensorboard
        # if use_tensorboard_epoch != 0.1 and self.rank in [-1,0]:
        #     self.writer = SummaryWriter(self.visualization_path)

        # Set DataLoader
        # if self.dataloaderX:
        #     self.train_loader = DataLoaderX(dataset=self.train_dataset, sampler=self.train_sampler,
        #                                     batch_size=self.trainer_batch_size, num_workers=self.num_workers,
        #                                     pin_memory=self.pin_memory)
        #     self.valid_loader = DataLoaderX(dataset=self.valid_dataset, sampler=self.valid_sampler,
        #                                     batch_size=self.trainer_batch_size, num_workers=self.num_workers,
        #                                     pin_memory=self.pin_memory)
        #     if self.test_dataset and self.test_sampler:
        #         self.test_loader = DataLoaderX(dataset=self.test_dataset, sampler=self.test_sampler,
        #                                        batch_size=self.trainer_batch_size, num_workers=self.num_workers,
        #                                        pin_memory=self.pin_memory)
        # else:
        #     self.train_loader = Data.DataLoader(dataset=self.train_dataset, sampler=self.train_sampler,
        #                                         batch_size=self.trainer_batch_size, num_workers=self.num_workers,
        #                                         pin_memory=self.pin_memory)
        #     self.valid_loader = Data.DataLoader(dataset=self.valid_dataset, sampler=self.valid_sampler,
        #                                         batch_size=self.trainer_batch_size, num_workers=self.num_workers,
        #                                         pin_memory=self.pin_memory)
        #     if self.test_dataset and self.test_sampler:
        #         self.test_loader = Data.DataLoader(dataset=self.test_dataset, sampler=self.test_sampler,
        #                                            batch_size=self.trainer_batch_size, num_workers=self.num_workers,
        #                                            pin_memory=self.pin_memory)

        # Set Metric
        if self.metric and self.rank in [-1,0]:
            self.metric.initialize(device=self.device,
                                   total_epoch=1,
                                   metric_type="valid",
                                   node_dict_len=len(self.lookuptable_E),
                                   model_name=self.model_name,
                                   # logger=self.logger,
                                   # writer=self.writer,
                                   train_dataset=self.train_dataset,
                                   valid_dataset=self.valid_dataset,
                                   test_dataset=self.test_dataset)
            if self.metric.link_prediction_filt:
                self.metric.establish_correct_triplets_dict()

        # Set Multi GPU
    def evaluate(self):
        # if self.total_epoch <= self.trained_epoch:
        #     raise ValueError("Trained_epoch is bigger than total_epoch!")

        # for epoch in range(self.total_epoch - self.trained_epoch):
        #     self.current_epoch = epoch + 1 + self.trained_epoch
        #
        #     # Train Progress
        #     train_epoch_loss = 0.0
        #     if self.rank == -1:
        #         self.model.train()
        #         for train_step, batch in enumerate(tqdm(self.train_loader)):
        #             train_loss = self.model.loss(batch)
        #             train_epoch_loss += train_loss.item()
        #             self.optimizer.zero_grad()
        #             if self.apex:
        #                 from apex import amp
        #                 with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
        #                     scaled_loss.backward()
        #             else:
        #                 train_loss.backward()
        #             self.optimizer.step()
        #     else:
        #         # self.logger.info("hahahahahah  Epoch:{}".format(self.current_epoch))
        #         self.train_sampler.set_epoch(epoch)
        #         self.model.train()
        #         for train_step, batch in enumerate(self.train_loader):
        #             train_loss = self.model.module.loss(batch)
        #             train_epoch_loss += reduce_mean(train_loss,dist.get_world_size()).item()
        #             self.optimizer.zero_grad()
        #             if self.apex:
        #                 from apex import amp
        #                 with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
        #                     scaled_loss.backward()
        #             else:
        #                 train_loss.backward()
        #             self.optimizer.step()
        #
        #     # Valid Process
        #     if self.rank in [-1,0]:
        #         with torch.no_grad():
        #             valid_epoch_loss = 0.0
        #             valid_model = self.model if self.rank == -1 else self.model.module
        #             valid_model.eval()
        #             for batch in self.valid_loader:
        #                 valid_loss = valid_model.loss(batch)
        #                 valid_epoch_loss += valid_loss.item()
        #                 # break
        #             average_train_epoch_loss = train_epoch_loss / len(self.train_dataset)
        #             average_valid_epoch_loss = valid_epoch_loss / len(self.valid_dataset)
        #             self.average_train_epoch_loss_list.append(average_train_epoch_loss)
        #             self.average_valid_epoch_loss_list.append(average_valid_epoch_loss)
        #             self.current_epoch_list.append(self.current_epoch)
        #             self.logger.info("Epoch{}/{}   Train Loss: {}   Valid Loss: {}".format(self.current_epoch,
        #                                                                         self.total_epoch,
        #                                                                         average_train_epoch_loss,
        #                                                                         average_valid_epoch_loss))
        #
        #     # Metric Progress
        #     if self.rank in [-1,0]:
        #         if self.use_metric_epoch and self.current_epoch % self.use_metric_epoch == 0:
        #             self.use_metric()
        #         # Tensorboard Process
        #         if self.current_epoch % self.use_tensorboard_epoch == 0:
        #             self.use_tensorboard(average_train_epoch_loss, average_valid_epoch_loss)
        #         # Savemodel Process
        #         if self.current_epoch % self.use_savemodel_epoch == 0 or (self.current_epoch!=0.1 and self.current_epoch==self.total_epoch):
        #             self.use_savemodel()
        #         # Matlotlib Process
        #         if self.current_epoch % self.use_matplotlib_epoch == 0:
        #             self.use_matplotlib()

        if self.rank in [-1,0]:
            # Summary Train Progress
            # self.summary_final_metric()
            self.evaluate_on_test_dataset()

    def use_metric(self):
        print("Evaluating Model {} on Valid Dataset...".format(self.model_name))
        valid_model = self.model.module if self.rank == 0 else self.model
        valid_model.eval()
        self.metric.caculate(model=valid_model, current_epoch=self.current_epoch)
        self.metric.print_current_table()
        self.metric.log()
        self.metric.write()
        print("-----------------------------------------------------------------------")
        if self.metric.current_model_is_better(self.best_metric) or self.best_metric == None:
            checkpoint_path=os.path.join(self.output_path, "checkpoints","best_model_{}".format(self.model_name))
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.logger.info("save {} epoch model as best model".format(self.current_epoch))
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "Model.pkl"))
            self.best_metric = self.metric.get_current_metric()
            self.best_epoch=self.current_epoch

    def use_tensorboard(self, average_train_epoch_loss, average_valid_epoch_loss):
        self.writer.add_scalars("Loss", {"train_loss": average_train_epoch_loss,
                                         "valid_loss": average_valid_epoch_loss},
                                self.current_epoch)

    def use_savemodel(self):
        checkpoint_path = os.path.join(self.output_path, "checkpoints",
                                       "{}_{}epochs".format(self.model_name, self.current_epoch))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        model = self.model.module if self.rank == 0 else self.model
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "Model.pkl"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "Optimizer.pkl"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_path, "Lr_Scheduler.pkl"))

    def use_matplotlib(self):
        plt.figure()
        plt.plot(np.array(self.current_epoch_list), np.array(self.average_train_epoch_loss_list),
                 label="train_loss",
                 color="cornflowerblue",
                 linewidth=4, linestyle="-")
        plt.plot(np.array(self.current_epoch_list), np.array(self.average_valid_epoch_loss_list),
                 label="valid_loss",
                 color="darkviolet",
                 linewidth=4, linestyle="--")
        plt.yscale('log')
        plt.legend(loc="upper right")
        plt.show()

    def summary_final_metric(self):
        self.metric.print_best_table()

    def evaluate_on_test_dataset(self):
        if self.test_dataset and self.test_sampler:
            self.metric.metric_type="test"
            # self.logger.info("Evaluating Model {} on Test Dataset...".format(self.model_name))
            # self.logger.info("Select {} epoch model to evaluate on test dataset".format(self.best_epoch))
            # self.model.load_state_dict(torch.load(os.path.join(os.path.join(self.output_path, "checkpoints","best_model_{}".format(self.model_name)), "Model.pkl")))
            test_model = self.model.module if self.rank == 0 else self.model
            test_model.eval()
            self.metric.caculate(model=test_model, current_epoch=1)
            self.metric.total_epoch=0
            self.metric.current_epoch=0
            self.metric.print_current_table()
            # self.metric.log()
            # self.metric.write()
            # print("-----------------------------------------------------------------------")




