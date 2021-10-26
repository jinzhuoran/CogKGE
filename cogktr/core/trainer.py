import numpy as np
import torch
import matplotlib.pyplot as plt
from .log import *
class Trainer:
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 model,
                 metric,
                 loss,
                 optimizer,
                 epoch,
                 save_step=None
                 ):
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.model=model
        self.metric=metric
        self.loss=loss
        self.optimizer=optimizer
        self.epoch=epoch
        self.save_step=save_step

        self.train_loss_list=list()
        self.valid_loss_list=list()

    def train(self):
        print("The training process is beginning!")
        self.model=self.model.cuda()
        for epoch in range(self.epoch):
            for step,train_batch in enumerate(self.train_dataset):
                train_batch=train_batch.cuda()
                random_valid_batch_idx=np.random.randint(len(self.valid_dataset)-1)
                for step_valid,valid_batch_temp in enumerate(self.valid_dataset):
                    if step_valid==random_valid_batch_idx:
                        valid_batch=valid_batch_temp
                        break
                valid_batch=valid_batch.cuda()



                output=self.model (train_batch)
                train_loss=self.loss(train_batch,self.model)
                valid_loss=self.loss(valid_batch,self.model)

                self.train_loss_list.append(train_loss.data.cpu().numpy())
                self.valid_loss_list.append(valid_loss.data.cpu().numpy())

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                if step % 300 == 0:
                    print('Epoch: ', epoch,
                          '| train loss: %.4f' % train_loss.data.cpu().numpy(),
                          '| valid loss: %.4f' % valid_loss.data.cpu().numpy())

            if self.save_step==None:
                pass
            else:
                if epoch % self.save_step == 0:
                    torch.save(self.model,"TransE_Model_%depochs.pkl"%(epoch))
                    logger.info("The model named \"%s_Model_%depochs.pkl\" has been saved!"%(self.model.name,epoch))

        torch.save(self.model,"TransE_Model_%depochs.pkl"%(self.epoch))
        logger.info("The model named \"%s_Model_%depochs.pkl\" has been saved!"%(self.model.name,self.epoch))

        print("The training process is finished!")

        return 0

    def show(self):
        print("The show process is beginning!")
        x=np.arange(0,len(self.train_loss_list))

        plt.figure()

        plt.plot(x,self.train_loss_list,color="red",linewidth=4,linestyle="-",label="train_loss")
        plt.legend(loc="upper right")
        plt.plot(x,self.valid_loss_list,color="blue",linewidth=4,linestyle="-",label="valid_loss")
        plt.legend(loc="upper right")

        plt.title("%s_loss_figure"%(self.model.name))
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.savefig(fname="%s_Model_%depochs_loss.png"%(self.model.name,self.epoch))
        # print("The picture named \"%s_Model_%depochs_loss.png\" has been saved!"%(self.model.name,self.epoch))
        logger.info("The picture named \"%s_Model_%depochs_loss.png\" has been saved!"%(self.model.name,self.epoch))

        plt.show()

        print("The show process is finished!")
        return 0
