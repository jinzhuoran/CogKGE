import torch
from torch.utils.data import Dataset


class Cog_Dataset(Dataset):
    def __init__(self, data, task, descriptions=None):
        """

        :param data: numpy array  (len,5) or (len,3)
        :param task: kr tr or ktr  currently only kr are supported
        """
        self.data = data
        self.task = task
        self.descriptions = descriptions
        self.data_name = 'dataset'

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.task == 'kr':
            if not self.descriptions:
                return torch.tensor(self.data[index], dtype=torch.long)
            else:
                return [torch.tensor(self.data[index], dtype=torch.long), *[
                    self.descriptions[i][index] for i in range(len(self.descriptions))
                ]]

        else:
            raise ValueError("{} currently are not supported!".format(self.task))
