from torch.utils.data import Dataset
import torch
import numpy as np
import torch.utils.data as Data
class Cog_Dataset(Dataset):
    def __init__(self,data,task):
        self.data=data
        self.task=task
        self.data_numpy=None
        self._datable2numpy()
    def __getitem__(self,index):
        if self.task=="kr":
            #把datable格式的data，在kr任务下是转换成numpy，再转tensor，再编序号
            return torch.tensor(self.data_numpy[index],dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def _datable2numpy(self):
        head_list=self.data["head"]
        relation_list=self.data["relation"]
        tail_list=self.data["tail"]
        head_np=np.array(head_list)[:,np.newaxis]
        relation_np=np.array(relation_list)[:,np.newaxis]
        tail_np=np.array(tail_list)[:,np.newaxis]
        self.data_numpy=np.hstack((head_np,relation_np,tail_np))

if __name__=="__main__":
    import prettytable as pt
    from datable import Datable
    #kr任务
    data = [["小A", "小B", "小C"],
            ["在", "去", "在"],
            ["家", "超市", "学校"]]
    dic_E={"小A":0,
           "小B":1,
           "小C":2,
           "家":3,
           "超市":4,
           "学校":5 }
    dic_R={"在":0,
           "去":1}

    datable = Datable()                                                       # 标准实例化
    datable(["head", "relation", "tail"], [data[0], data[1], data[2]])        # 标准添加数据方式二
    datable.print_table()                                                     # 标准打印datable方式
    #字符串转数字
    for i in range(len(datable)):
        datable["head"][i]=dic_E[datable["head"][i]]
        datable["relation"][i]=dic_R[datable["relation"][i]]
        datable["tail"][i]=dic_E[datable["tail"][i]]
    datable.print_table()
    #datable转dataset
    dataset=Cog_Dataset(data=datable,task="kr")
    print(len(dataset))
    print(dataset[1])
    #检查dataset是否创建合适
    data_loader = Data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False)
    for epoch in range(3):
        for step, b_x in enumerate(data_loader):
            print("b_x.shape",b_x.data.numpy().shape)
            print("epoch:",epoch,"step:",step,"\nb_x:\n",b_x)
