import torch
import torch.utils.data as Data
from tqdm import tqdm
import random
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Kr_Trainer:
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 train_sampler,
                 valid_sampler,
                 trainer_batch_size,
                 model,
                 loss,
                 optimizer,
                 metric,
                 epoch,
                 output_path,
                 save_step=None,
                 metric_step=None,
                 save_final_model=False,
                 visualization=False
                 ):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.trainer_batch_size = trainer_batch_size
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.epoch = epoch
        self.output_path = output_path
        self.save_step = save_step
        self.metric_step = metric_step
        self.save_final_model = save_final_model
        self.visualization = visualization

    def create_negative(self, train_pos):
        train_neg = None
        if self.model.negative_sample_method == "Random_Negative_Sampling":
            train_neg = train_pos.clone().detach()
            for i in range(len(train_neg[:, 0])):
                if (random.random() < 0.5):
                    train_neg[i][0] = np.random.randint(0, self.model.entity_dict_len)
                else:
                    train_neg[i][2] = np.random.randint(0, self.model.entity_dict_len)
        return train_neg

    def train(self):
        train_loader = Data.DataLoader(dataset=self.train_dataset, sampler=self.train_sampler,
                                       batch_size=self.trainer_batch_size)
        self.model = self.model.cuda()

        if self.visualization == True:
            if not os.path.exists(os.path.join(self.output_path, "visualization", self.model.name)):
                os.makedirs(os.path.join(self.output_path, "visualization", self.model.name))
            writer = SummaryWriter(os.path.join(self.output_path, "visualization", self.model.name))
            print("本次可视化的保存路径为", os.path.join(self.output_path, "visualization", self.model.name).replace('\\', '/'))
            print("cd到FB15k-237目录后，请输入", "tensorboard --logdir=experimental_output", "然后把网站地址输入浏览器")

        raw_meanrank = -1
        raw_hitatten = -1

        for epoch in range(self.epoch):
            with tqdm(train_loader, ncols=150) as t:
                for step, train_positive in enumerate(t):
                    train_positive = train_positive.cuda()
                    train_negative = self.create_negative(train_positive)
                    train_positive_embedding = self.model(train_positive)
                    train_negative_embedding = self.model(train_negative)
                    train_loss = self.loss(train_positive_embedding, train_negative_embedding)

                    valid_loader = Data.DataLoader(dataset=self.valid_dataset, sampler=self.valid_sampler,
                                                   batch_size=self.trainer_batch_size)
                    for step_valid, valid_positive in enumerate(valid_loader):
                        if step_valid == 0:
                            pass
                        else:
                            break
                    valid_positive = valid_positive.cuda()
                    valid_negative = self.create_negative(valid_positive)
                    valid_positive_embedding = self.model(valid_positive)
                    valid_negative_embedding = self.model(valid_negative)
                    valid_loss = self.loss(valid_positive_embedding, valid_negative_embedding)

                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()

                    t.set_description("ep%d|st%d" % (epoch, step))
                    t.set_postfix({'train_loss:': '%.2f' % (train_loss),
                                   'valid_loss:': '%.2f' % (valid_loss),
                                   'mr:': '%.1f' % (raw_meanrank),
                                   'hat:': '%.0f%%' % (raw_hitatten)})

                # 每隔几步评价模型
                if self.metric_step != None and epoch % self.metric_step == 0:
                    if self.metric.name == "Link_Prediction":
                        self.metric(self.model, self.valid_dataset)
                        raw_meanrank = self.metric.raw_meanrank
                        raw_hitatten = self.metric.raw_hitatten

            # 每轮的可视化
            if self.visualization == True:
                writer.add_scalars("1_loss", {"train_loss": train_loss,
                                              "valid_loss": valid_loss}, epoch)
                if self.metric.name == "Link_Prediction":
                    writer.add_scalars("2_meanrank", {"valid_raw_meanrank": raw_meanrank}, epoch)
                    writer.add_scalars("3_hitatten", {"valid_raw_hitatten": raw_hitatten}, epoch)
                if epoch == 0:
                    fake_data = torch.zeros((len(self.train_dataset), 3)).long().cuda()
                    writer.add_graph(self.model, fake_data)
                for name, param in self.model.named_parameters():
                    writer.add_histogram(name + '_grad', param.grad, epoch)
                    writer.add_histogram(name + '_data', param, epoch)
                if epoch == 0:
                    embedding_data = torch.rand(10, 20)
                    embedding_label = ["篮球", "足球", "乒乓球", "羽毛球", "保龄球", "游泳", "爬山", "旅游", "赛车", "写代码"]
                    writer.add_embedding(mat=embedding_data, metadata=embedding_label)

            # 每隔几步保存模型
            if self.save_step != None and epoch % self.save_step == 0:
                if not os.path.exists(os.path.join(self.output_path, "checkpoints")):
                    os.makedirs(os.path.join(self.output_path, "checkpoints"))
                torch.save(self.model, os.path.join(self.output_path, "checkpoints",
                                                    "%s_Model_%depochs.pkl" % (self.model.name, epoch)))

        t.close()

        # 保存最终模型
        if self.save_final_model == True:
            if not os.path.exists(os.path.join(self.output_path, "checkpoints")):
                os.makedirs(os.path.join(self.output_path, "checkpoints"))
                print(os.path.join(self.output_path, "checkpoints") + ' created successfully')
            torch.save(self.model, os.path.join(self.output_path, "checkpoints",
                                                "%s_Model_%depochs.pkl" % (self.model.name, self.epoch)))
            print(
                os.path.join(self.output_path, "checkpoints", "%s_Model_%depochs.pkl" % (self.model.name, self.epoch)),
                "saved successfully")

        pass
