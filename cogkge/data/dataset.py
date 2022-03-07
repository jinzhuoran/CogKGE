import torch
from torch.utils.data import Dataset


class Cog_Dataset(Dataset):
    def __init__(self, data, task, descriptions=None,train_pattern="score_based"):
        """

        :param data: numpy array  (len,5) or (len,3)
        :param task: kr tr or ktr  currently only kr are supported
        """
        self.data = data
        self.task = task
        self.descriptions = descriptions
        self.data_name = 'dataset'
        self.train_pattern=train_pattern

    def __len__(self):
        if self.train_pattern == "classification_based":
            return self.data[0].shape[0]
        if self.train_pattern == "score_based":
            return self.data.shape[0]

    def __getitem__(self, index):
        if self.task == 'kr':
            if self.train_pattern =="classification_based":
                return torch.tensor(self.data[0][index],dtype=torch.long),\
                       torch.tensor(self.data[1][index],dtype=torch.long)

            if self.train_pattern =="score_based":
                if not self.descriptions:
                    return torch.tensor(self.data[index], dtype=torch.long)
                else:
                    return [torch.tensor(self.data[index], dtype=torch.long), *[
                        self.descriptions[i][index] for i in range(len(self.descriptions))
                    ]]

        else:
            raise ValueError("{} currently are not supported!".format(self.task))

