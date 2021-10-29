from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch

class Feeder(Dataset):
    def __init__(self,data_numpy):
        self.data_numpy = data_numpy
    
    def __getitem__(self, index):
        return torch.tensor(self.data_numpy[index],dtype=torch.long)
    
    def __len__(self):
        return self.data_numpy.shape[0]
