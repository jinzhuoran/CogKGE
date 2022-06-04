import torch
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # CogKGE root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKGE root directory to PATH

from cogkge import *

device = init_cogkge(device_id="5", seed=1)

loader = WIKIPEDIA5MLoader(dataset_path="../dataset", download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut = loader.load_all_lut()

model = TransE(entity_dict_len=len(node_lut),
               relation_dict_len=len(relation_lut),
               embedding_dim=100,
               p_norm=1)
model.load_state_dict(torch.load(
    "/data/mentianyi/code/CogKGE/dataset/WIKIPEDIA5M/experimental_output/TransE2022-06-04--19-20-22.14--500epochs/checkpoints/TransE_100epochs/Model.pkl"))
model = model.to("cuda:0")
result = model.e_embedding(torch.tensor(node_lut.vocab.word2idx[18978754]).to("cuda:0"))

# predictor = Predictor(
#     model=model,
#     pretrained_model_path="/data/mentianyi/code/CogKGE/dataset/WIKIPEDIA5M/experimental_output/TransE2022-06-04--19-20-22.14--500epochs/checkpoints/TransE_40epochs/Model.pkl",
#     model_name="TransE",
#     data_name="WIKIPEDIA5M",
#     device="cuda",
#     node_lut=node_lut,
#     relation_lut=relation_lut,
#     processed_data_path="../dataset",)
# print("end")
