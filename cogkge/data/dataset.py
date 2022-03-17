import torch
from torch.utils.data import Dataset


class Cog_Dataset(Dataset):
    def __init__(self, data, task, mode=None,descriptions=None,train_pattern="score_based",
                 lookuptable_E=None,lookuptable_R=None,
                 node_type=False,relation_type=False,time=False):
        """
        :param data: numpy array  (len,5) or (len,3)
        :param task: kr tr or ktr  currently only kr are supported
        """
        self.mode = mode
        self.label_data = None
        if isinstance(data,tuple):
            self.label_data = torch.tensor(data[1],dtype=torch.float)
            self.data = data[0]
        else:
            self.data = data
        self.task = task
        self.descriptions = descriptions
        self.data_name = 'dataset'
        self.train_pattern=train_pattern
        self.lookuptable_E = lookuptable_E
        self.lookuptable_R = lookuptable_R
        self.node_type = node_type
        self.relation_type = relation_type
        self.time = time

    def __len__(self):
        return self.data.shape[0]

    def update_sample(self,sample,index):
        if self.lookuptable_E:
            if self.node_type:
                sample.update({"h_type": self.lookuptable_E.type[self.data[index][0]],
                               "t_type": self.lookuptable_E.type[self.data[index][2]]})
            if self.descriptions:
                sample.update({"h_token": self.lookuptable_E.token[self.data[index][0]],
                               "t_token": self.lookuptable_E.token[self.data[index][2]],
                               "h_mask": self.lookuptable_E.mask[self.data[index][0]],
                               "t_mask": self.lookuptable_E.mask[self.data[index][2]]})
        if self.lookuptable_R:
            if self.relation_type:
                sample.update({"r_type": self.lookuptable_R.type[self.data[index][1]]})

        if self.time:
            sample.update({"start":self.data[index][3],
                            "end":self.data[index][4]})
        sample = tuple(sample.values())
        return sample

    def __getitem__(self,index):
        if self.mode == "type":
            return (self.data[index][0],
                    self.data[index][1],
                    self.data[index][2],)
        elif self.mode == "description":
            pass
        elif self.mode == "normal":
            return (self.data[index][0],
                    self.data[index][1],
                    self.data[index][2],)
        elif self.mode == "time":
            pass
        else:
            raise ValueError("{} mode not supported!".format(self.mode))


    # def __getitem__(self, index):
    #     if self.task == 'kr':
    #         sample = {}
    #         sample.update({"h": self.data[index][0],
    #                        "r": self.data[index][1],})
    #         if self.train_pattern == "classification_based":
    #             sample["t"] = self.label_data[index]
    #         else:
    #             sample["t"] = self.data[index][2]
    #         return self.update_sample(sample,index)
    #     else:
    #         raise ValueError("{} currently are not supported!".format(self.task))

