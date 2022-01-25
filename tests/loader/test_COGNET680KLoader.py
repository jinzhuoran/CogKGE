import torch
from torch.utils.data import RandomSampler
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0] # CogKGE root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKGE root directory to PATH


from cogkge import *
import time


print("Testing COGNET680K dataloader...")
loader = COGNET680KLoader(dataset_path='/data/hongbang/CogKGE/dataset/',
                    download=True,
                    )

start_time = time.time()
train_data, valid_data, test_data = loader.load_all_data()
node_lut,relation_lut = loader.load_all_lut()

processor = COGNET680KProcessor(node_lut, relation_lut)

train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)

for i in range(2):
    print(train_dataset[i])
    print(valid_dataset[i])
    print(test_dataset[i])
print("Train:{}  Valid:{} Test:{}".format(len(train_dataset),
                                          len(valid_dataset),
                                          len(test_dataset)))
print("--- %s seconds ---" % (time.time() - start_time))



