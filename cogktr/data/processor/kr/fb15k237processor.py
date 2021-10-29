from ...loader import *
import numpy as np
from ...lut import LUT
from ...dataset import Feeder

class FB15K237Processor:
    def __init__(self,lookUpTable:LUT):
        self.lut = lookUpTable
    
    def process(self,data):
        """
        convert list of string triples to corresponding datasets
        return: constructed dataset containing __len__ and __getitem__ methods
        """
        return Feeder(self.list2numpy(data))
        

    def list2numpy(self,data):
        """
        data: triples in string form(entity names and relation names)
        return: triples in numpy form
        """
        heads=list()
        relations=list()
        tails=list()

        for i in range(len(data[0])):
            heads.append(self.lut.entity2id(data[0][i]))
            relations.append(self.lut.relation2id(data[1][i]))
            tails.append(self.lut.entity2id(data[2][i]))

        heads_np=np.array(heads,dtype=np.int64)[:,np.newaxis]
        relations_np=np.array(relations,dtype=np.int64)[:,np.newaxis]
        tails_np=np.array(tails,dtype=np.int64)[:,np.newaxis]
        data_numpy=np.hstack((heads_np,relations_np,tails_np))

        return data_numpy
 
