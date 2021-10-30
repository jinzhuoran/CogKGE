# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from .log import *
# from .evaluator import *
# import math
# class Trainer:
#     def __init__(self,
#                  train_dataset,
#                  valid_dataset,
#                  model,
#                  metric,
#                  loss,
#                  optimizer,
#                  epoch,
#                  save_step=None,
#                  metric_step=None
#                  ):
#         self.train_dataset=train_dataset
#         self.valid_dataset=valid_dataset
#         self.model=model
#         self.metric=metric
#         self.loss=loss
#         self.optimizer=optimizer
#         self.epoch=epoch
#         self.save_step=save_step
#         self.metric_step=metric_step
#
#         self.train_loss_list=list()
#         self.valid_loss_list=list()
#         self.step_metric_list=list()
#         self.train_metric_result_rank_numpy_list=list()
#         self.valid_metric_result_rank_numpy_list=list()
#         self.train_metric_mean_rank_list=list()
#         self.valid_metric_mean_rank_list=list()
#         self.train_metric_hit_at_ten_list=list()
#         self.valid_metric_hit_at_ten_list=list()
#
#     def train(self):
#         print("The training process is beginning!")
#         self.model=self.model.cuda()
#         for epoch in range(self.epoch):
#             for step,train_batch in enumerate(self.train_dataset):
#                 train_batch=train_batch.cuda()
#                 random_valid_batch_idx=np.random.randint(len(self.valid_dataset)-1)
#                 for step_valid,valid_batch_temp in enumerate(self.valid_dataset):
#                     if step_valid==random_valid_batch_idx:
#                         valid_batch=valid_batch_temp
#                         break
#                 valid_batch=valid_batch.cuda()
#
#
#
#                 output=self.model (train_batch)
#                 train_loss=self.loss(train_batch,self.model)
#                 valid_loss=self.loss(valid_batch,self.model)
#
#                 self.train_loss_list.append(train_loss.data.cpu().numpy())
#                 self.valid_loss_list.append(valid_loss.data.cpu().numpy())
#
#                 self.optimizer.zero_grad()
#                 train_loss.backward()
#                 self.optimizer.step()
#
#                 if step % 300 == 0:
#                     print('Epoch: ', epoch,
#                           '| train loss: %.4f' % train_loss.data.cpu().numpy(),
#                           '| valid loss: %.4f' % valid_loss.data.cpu().numpy())
#
#             if self.save_step==None:
#                 pass
#             else:
#                 if epoch % self.save_step == 0:
#                     torch.save(self.model,"TransE_Model_%depochs.pkl"%(epoch))
#                     logger.info("The model named \"%s_Model_%depochs.pkl\" has been saved!"%(self.model.name,epoch))
#
#             if self.metric_step==None:
#                 pass
#             else:
#                 if epoch % self.metric_step == 0:
#                     torch.save(self.model,"TransE_Model_temp.pkl")
#
#                     evaluator_train=Evaluator(
#                         test_dataset=self.train_dataset,
#                         model_path="TransE_Model_temp.pkl",
#                         metric=self.metric
#                     )
#                     evaluator_train.evaluate()
#                     evaluator_valid=Evaluator(
#                         test_dataset=self.valid_dataset,
#                         model_path="TransE_Model_temp.pkl",
#                         metric=self.metric
#                     )
#                     evaluator_valid.evaluate()
#
#                     self.step_metric_list.append(epoch)
#                     self.train_metric_result_rank_numpy_list.append(evaluator_train.result_rank_numpy)
#                     self.valid_metric_result_rank_numpy_list.append(evaluator_valid.result_rank_numpy)
#                     self.train_metric_mean_rank_list.append(evaluator_train.mean_rank )
#                     self.valid_metric_mean_rank_list.append(evaluator_valid.mean_rank )
#                     self.train_metric_hit_at_ten_list.append(evaluator_train.hit_at_ten )
#                     self.valid_metric_hit_at_ten_list.append(evaluator_valid.hit_at_ten )
#
#
#         torch.save(self.model,"TransE_Model_%depochs.pkl"%(self.epoch))
#         logger.info("The model named \"%s_Model_%depochs.pkl\" has been saved!"%(self.model.name,self.epoch))
#
#         print("The training process is finished!")
#
#         return 0
#
#     def show(self):
#         print("The show process is beginning!")
#         x=np.arange(0,len(self.train_loss_list))
#
#         plt.figure(figsize=(10,10))
#
#         plt.subplot(2,2,1)
#         plt.plot(x,self.train_loss_list,color="red",linewidth=4,linestyle="-",label="train_loss")
#         plt.legend(loc="upper right")
#         plt.plot(x,self.valid_loss_list,color="blue",linewidth=4,linestyle="-",label="valid_loss")
#         plt.legend(loc="upper right")
#         plt.title("%s_loss_figure"%(self.model.name))
#         plt.xlabel("step")
#         plt.ylabel("loss")
#
#         plt.subplot(2,2,2)
#         plt.plot(self.step_metric_list,self.train_metric_mean_rank_list,color="red",linewidth=4,linestyle="-",label="train_mean_rank")
#         plt.legend(loc="upper right")
#         plt.plot(self.step_metric_list,self.valid_metric_mean_rank_list,color="blue",linewidth=4,linestyle="-",label="valid_mean_rank")
#         plt.legend(loc="upper right")
#         plt.title("%s_mean_rank_figure"%(self.model.name))
#         plt.xlabel("epoch")
#         plt.ylabel("rank")
#
#         plt.subplot(2,2,3)
#         plt.plot(self.step_metric_list,self.train_metric_hit_at_ten_list,color="red",linewidth=4,linestyle="-",label="train_hit_at_ten")
#         plt.legend(loc="upper right")
#         plt.plot(self.step_metric_list,self.valid_metric_hit_at_ten_list,color="blue",linewidth=4,linestyle="-",label="valid_hit_at_ten")
#         plt.legend(loc="upper right")
#         plt.title("%s_hit_at_ten_figure"%(self.model.name))
#         plt.xlabel("epoch")
#         plt.ylabel("rank percent(%%)")
#
#         plt.subplot(2,2,4)
#
#         length_result_rank_numpy=len(self.train_metric_result_rank_numpy_list[-1])
#         length_new=math.floor(length_result_rank_numpy**0.5)
#         plt.imshow(self.train_metric_result_rank_numpy_list[-1][0:length_new**2].reshape((length_new,length_new)),cmap="Blues",origin="upper")
#         plt.colorbar(shrink=0.9)
#         plt.title("%s_rank_array_figure"%(self.model.name))
#
#         plt.savefig(fname="%s_Model_%depochs_figure.png"%(self.model.name,self.epoch))
#         logger.info("The picture named \"%s_Model_%depochs_figure.png\" has been saved!"%(self.model.name,self.epoch))
#
#         plt.show()
#
#         print("The show process is finished!")
#         return 0


########################################################################################################################
import torch.utils.data as Data
from tqdm import tqdm
class Trainer:
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 train_sampler,
                 valid_sampler,
                 trainer_batch_size,
                 model,
                 loss,
                 optimizer,
                 epoch
                 ):
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.train_sampler=train_sampler
        self.valid_sampler=valid_sampler
        self.trainer_batch_size=trainer_batch_size
        self.model=model
        self.loss=loss
        self.optimizer=optimizer
        self.epoch=epoch


    def train(self):
        train_loader = Data.DataLoader(dataset=self.train_dataset,sampler=self.train_sampler,batch_size=self.trainer_batch_size)
        valid_loader = Data.DataLoader(dataset=self.valid_dataset,sampler=self.valid_sampler,batch_size=self.trainer_batch_size)
        self.model=self.model.cuda()

        for epoch in range(self.epoch):
            with tqdm(train_loader) as t:
                for step,train_batch in enumerate(t):
                    train_batch=train_batch.cuda()
                    output=self.model(train_batch)
                    loss=self.loss(output,train_batch,self.model)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()



                    t.set_description("epoch %d|step %d"%(epoch,step))
                    t.set_postfix({'train loss: ' : '%.2f'%(loss),
                           'test accuracy: ':'%.2f'%(1)})
                    pass
        t.close()

        pass

