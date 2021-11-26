from logging import Logger
import torch
from torch.nn import parallel
import torch.utils.data as Data
from tqdm import tqdm
import random
import os
import numpy as np

from cogktr.data import dataset
from torch.utils.tensorboard import SummaryWriter
from .log import *


class Kr_Trainer:
    def __init__(self,
                 logger,
                 train_dataset,
                 valid_dataset,
                 train_sampler,
                 valid_sampler,
                 negative_sampler,
                 trainer_batch_size,
                 model,
                 loss,
                 optimizer,
                 metric,
                 epoch,
                 output_path,
                 device,
                 lr_scheduler,
                 lookuptable_E=None,
                 lookuptable_R=None,
                 save_step=None,
                 metric_step=None,
                 save_final_model=False,
                 visualization=False,
                 ):
        self.logger = logger
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lookuptable_E=lookuptable_E
        self.lookuptable_R=lookuptable_R
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.negative_sampler = negative_sampler
        self.trainer_batch_size = trainer_batch_size
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.epoch = epoch
        self.output_path = output_path
        self.device=device
        self.save_step = save_step
        self.metric_step = metric_step
        self.save_final_model = save_final_model
        self.visualization = visualization
        self.lr_scheduler = lr_scheduler

    def train(self):
        train_loader = Data.DataLoader(dataset=self.train_dataset, sampler=self.train_sampler,
                                       batch_size=self.trainer_batch_size)
        valid_loader = Data.DataLoader(dataset=self.valid_dataset,sampler=self.valid_sampler,
                                        batch_size=self.trainer_batch_size)
        if self.lookuptable_E!=None:
            self.model.load_lookuotable(self.lookuptable_E,self.lookuptable_R,self.device)
        self.model = self.model.to(self.device)
        print("Available cuda devices:",torch.cuda.device_count())
        parallel_model = torch.nn.DataParallel(self.model)

        if self.visualization == True:
            if not os.path.exists(os.path.join(self.output_path, "visualization", self.model.name)):
                os.makedirs(os.path.join(self.output_path, "visualization", self.model.name))
            writer = SummaryWriter(os.path.join(self.output_path, "visualization", self.model.name))
            self.logger.info("The visualization path is"+os.path.join(self.output_path, "visualization", self.model.name).replace('\\', '/'))
            self.logger.info("After cd FB15k-237 dir，please enter tensorboard --logdir=experimental_output")
            self.logger.info("Enter the website address into the browser")

        raw_meanrank = -1
        raw_hitatten = -1
        
        mean_ranks = []
        mrrs = []
        hitattens = []


        for epoch in range(self.epoch):
            # Training Progress
            epoch_loss = 0.0
            for train_step, train_positive in enumerate(tqdm(train_loader)):
                train_positive = train_positive.to(self.device)
                train_negative = self.negative_sampler.create_negative(train_positive)
                train_positive_score = parallel_model(train_positive)
                train_negative_score = parallel_model(train_negative)
                train_loss = self.loss(train_positive_score, train_negative_score)
                
                self.optimizer.zero_grad()
                train_loss.backward()
                epoch_loss = epoch_loss + train_loss.item()
                self.optimizer.step()

            valid_epoch_loss = 0.0
            with torch.no_grad():
                for valid_step,valid_positive in enumerate(valid_loader):
                    valid_positive = valid_positive.to(self.device)
                    valid_negative = self.negative_sampler.create_negative(valid_positive)
                    valid_positive_score = parallel_model(valid_positive)
                    valid_negative_score = parallel_model(valid_negative)
                    valid_loss = self.loss(valid_positive_score,valid_negative_score)

                    valid_epoch_loss = valid_epoch_loss + valid_loss.item()

            print("Epoch{}/{}   Train Loss:".format(epoch+1,self.epoch),epoch_loss/(train_step+1),
                                                    " Valid Loss:",valid_epoch_loss/(valid_step+1))

            # self.lr_scheduler.step(valid_epoch_loss/(valid_step+1))
            # self.lr_scheduler.step()
            # 每隔几步评价模型
            if self.metric_step != None and (epoch+1) % self.metric_step == 0:
                if self.metric.name == "Link_Prediction":
                    print("Evaluating Model {}...".format(self.model.name))
                    # self.metric(self.model, self.valid_dataset,self.device)
                    self.metric(parallel_model,self.valid_dataset,self.device)
                    raw_meanrank = self.metric.raw_meanrank
                    raw_hitatten = self.metric.raw_hitatten
                    raw_mrr = self.metric.raw_MRR
                    print("mean rank:{}     hit@10:{}   MRR:{}".format(raw_meanrank,raw_hitatten,raw_mrr))
                    self.logger.info("Epoch {}/{}  mean_rank:{}   hit@10:{}   MRR:{}".format(
                        epoch+1,self.epoch,raw_meanrank,raw_hitatten,raw_mrr
                    ))
                    self.lr_scheduler.step(raw_meanrank)
                    mean_ranks.append([[raw_meanrank,epoch+1]])
                    hitattens.append([[raw_hitatten,epoch+1]])
                    mrrs.append([[raw_mrr,epoch+1]])
                    print("-----------------------------------------------------------------------")
                    if self.visualization == True:
                        writer.add_scalars("2_meanrank", {"valid_raw_meanrank": raw_meanrank}, epoch+1)
                        writer.add_scalars("3_hitatten", {"valid_raw_hitatten": raw_hitatten}, epoch+1)
                        writer.add_scalars("4_MRR",{"valid_raw_MRR":raw_mrr},epoch+1)

            
            # Evaluation Process

            
            # 每轮的可视化
            if self.visualization == True:
                writer.add_scalars("1_loss", {"train_loss": train_loss,
                                              "valid_loss": valid_loss}, epoch+1)
                if epoch == 0:
                    fake_data = torch.zeros(self.trainer_batch_size, 3).long()
                    writer.add_graph(self.model.cpu(), fake_data)
                    self.model.to(self.device)
                # for name, param in self.model.named_parameters():
                #     writer.add_histogram(name + '_grad', param.grad, epoch)
                #     writer.add_histogram(name + '_data', param, epoch)
                # if epoch == 0:
                #     embedding_data = torch.rand(10, 20)
                #     embedding_label = ["篮球", "足球", "乒乓球", "羽毛球", "保龄球", "游泳", "爬山", "旅游", "赛车", "写代码"]
                #     writer.add_embedding(mat=embedding_data, metadata=embedding_label)

            # 每隔几步保存模型
            if self.save_step != None and (epoch+1) % self.save_step == 0:
                if not os.path.exists(os.path.join(self.output_path, "checkpoints")):
                    os.makedirs(os.path.join(self.output_path, "checkpoints"))
                torch.save(self.model, os.path.join(self.output_path, "checkpoints",
                                                    "%s_Model_%depochs.pkl" % (self.model.name, epoch+1)))

        # Record the top5 mean_rank and hit@10 in the log file:
        mean_ranks.sort(key=lambda x:x[0], reverse=False) # 1->2->3
        hitattens.sort(key=lambda x:x[0],reverse=True) # 3->2->1
        mrrs.sort(key=lambda x:x[0],reverse=True) # 3->2->1

        mean_ranks = mean_ranks if len(mean_ranks) < 5 else mean_ranks[:5]
        hitattens = hitattens if len(hitattens) < 5 else hitattens[:5]
        mrrs = mrrs if len(mrrs) < 5 else mrrs[:5]
        self.logger.info("Top Mean Rank: {}".format(mean_ranks))  
        self.logger.info("Top Hit@10: {}".format(hitattens)) 
        self.logger.info("Top MRR: {}".format(mrrs))
        print("Top Mean Rank: {}".format(mean_ranks))
        print("Top Hit@10: {}".format(hitattens)) 
        print("Top MRR: {}".format(mrrs))
    

        # 保存最终模型
        if self.save_final_model == True:
            if not os.path.exists(os.path.join(self.output_path, "checkpoints")):
                os.makedirs(os.path.join(self.output_path, "checkpoints"))
                self.logger.info(os.path.join(self.output_path, "checkpoints") + ' created successfully')
            torch.save(self.model, os.path.join(self.output_path, "checkpoints",
                                                "%s_Model_%depochs.pkl" % (self.model.name, self.epoch+1)))
            self.logger.info(
                os.path.join(self.output_path, "checkpoints", "%s_Model_%depochs.pkl" % (self.model.name, self.epoch+1))+
                "saved successfully")

        



            # with tqdm(train_loader, ncols=150) as t:

            
            # with tqdm(train_loader) as t:
            #     for step, train_positive in enumerate(t):
            #         train_positive = train_positive.to(self.device)
            #         train_negative = self.create_negative(train_positive)
            #         train_positive_embedding = self.model(train_positive)
            #         train_negative_embedding = self.model(train_negative)
            #         train_loss = self.loss(train_positive_embedding, train_negative_embedding)

            #         # valid_loader = Data.DataLoader(dataset=self.valid_dataset, sampler=self.valid_sampler,
            #         #                                batch_size=self.trainer_batch_size)
            #         # for step_valid, valid_positive in enumerate(valid_loader):
            #         #     if step_valid == 0:
            #         #         pass
            #         #     else:
            #         #         break
            #         # valid_positive = next(iter(valid_loader))
            #         # valid_positive = valid_positive.to(self.device)
            #         # valid_negative = self.create_negative(valid_positive)
            #         # valid_positive_embedding = self.model(valid_positive)
            #         # valid_negative_embedding = self.model(valid_negative)
            #         # valid_loss = self.loss(valid_positive_embedding, valid_negative_embedding)

            #         self.optimizer.zero_grad()
            #         train_loss.backward()
            #         self.optimizer.step()

            #         t.set_description("ep%d|st%d" % (epoch, step))
            #         t.set_postfix({'train_loss:': '%.2f' % (train_loss),
            #                        'valid_loss:': '%.2f' % (valid_loss),
            #                        'mr:': '%.1f' % (raw_meanrank),
            #                        'hat:': '%.0f%%' % (raw_hitatten)})

            #     # 每隔几步评价模型
            #     if self.metric_step != None and epoch % self.metric_step == 0:
            #         if self.metric.name == "Link_Prediction":
            #             self.metric(self.model, self.valid_dataset,self.device)
            #             raw_meanrank = self.metric.raw_meanrank
            #             raw_hitatten = self.metric.raw_hitatten
