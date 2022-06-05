import torch
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # CogKGE root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKGE root directory to PATH

from cogkge import *
from cogkge.data.lut import LookUpTable
import pickle

device = init_cogkge(device_id="5", seed=1)
with open("/data/mentianyi/code/CogKGE/dataset/WIKIPEDIA5M/processed_data/vocab.pkl", "rb") as file:
    node_vocab, relation_vocab =pickle.load(file)
node_lut = LookUpTable()
node_lut.add_vocab(node_vocab)
relation_lut = LookUpTable()
relation_lut.add_vocab(relation_vocab)

model = TransE(entity_dict_len=len(node_lut),
               relation_dict_len=len(relation_lut),
               embedding_dim=100,
               p_norm=1)
model.load_state_dict(torch.load(
    "/data/mentianyi/code/CogKGE/dataset/WIKIPEDIA5M/experimental_output/TransE2022-06-04--19-20-22.14--500epochs/checkpoints/TransE_100epochs/Model.pkl"))
model = model.to("cuda:0")
result = model.e_embedding(torch.tensor(node_lut.vocab.word2idx[18978754]).to("cuda:0"))
