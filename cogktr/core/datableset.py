from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch


class DataTableSet(Dataset):
    """
    create dataset object directly from the generated numpy file
    data_path:the output path of the preprocessing script,where the generated data file locates
    prefix:choose train,valid or test dataset
    debug:only use a small amount of data to test the whole functionality
    """
    def __init__(self,data_path,prefix,debug=False):
        if prefix not in ['train','valid','test']:
            raise ValueError("prefix argument must be train,valid or test but got {}!".format(prefix))
        self.data_path = data_path
        self.prefix = prefix
        self.debug = debug
        self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_path,'entity_set.pkl'),'rb') as f:
            self.entity_set = pickle.load(f)
        with open(os.path.join(self.data_path,'relation_set.pkl'),'rb') as f:
            self.relation_set = pickle.load(f)

        self.triples = np.load(os.path.join(self.data_path,'{}_triples.npy'.format(self.prefix)),mmap_mode='r')

        if self.debug:
            self.triples = self.triples[0:20]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        pos_sample =self.triples[index]
        return torch.tensor(pos_sample,dtype=torch.long)

if __name__ == '__main__':
    print("Start testing the feeder class on directory{} ...".format(os.getcwd()))

    data_path = '../../dataset/FB15k-237'
    for prefix in ['train','test','valid']:
        print("{} dataset:".format(prefix))
        loader = torch.utils.data.DataLoader(
            dataset=DataTableSet(data_path,prefix,debug=True),
            batch_size=7,
            shuffle=False)
        for i,batch in enumerate(loader):
            print(i,"  type(batch):",type(batch),"batch.shape:",batch.shape,)
    
    print("Testing finished!")


