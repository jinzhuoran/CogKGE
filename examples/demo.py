import os
import torch
import random
import argparse
import datetime
import numpy as np
import yaml
import shutil
from torch.utils.data import RandomSampler
from cogkge import *

loader = FB15K237Loader("../dataset/kr/FB15K237/raw_data", True, "Research_code/CogKTR/dataset")
train_data, valid_data, test_data = loader.load_all_data()
node_vocab, relation_vocab = loader.load_all_vocabs()
print("end")