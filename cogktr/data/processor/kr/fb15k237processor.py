from ...loader import *
import numpy as np

class FB15K237Processor:
    def __init__(self,path):
        self._path=path
        self._loader=FB15K237Loader(self._path)
        self._entity2idx=self._loader.load_entity_dict()
        self._relation2idx=self._loader.load_relation_dict()

    def process(self,data):
        heads=list()
        relations=list()
        tails=list()

        for i in range(len (data[0])):
            heads.append(self._entity2idx[data[0][i] ])
            relations.append(self._relation2idx[data[1][i] ])
            tails.append(self._entity2idx[data[2][i] ])

        heads_np=np.array(heads,dtype=np.int64)[:,np.newaxis]
        relations_np=np.array(relations,dtype=np.int64)[:,np.newaxis]
        tails_np=np.array(tails,dtype=np.int64)[:,np.newaxis]
        datable=np.hstack((heads_np,relations_np,tails_np))

        return datable