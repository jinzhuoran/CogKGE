# command:python -m torch.distributed.launch --nproc_per_node 2 test_ddp.py
# or choose specific gpus: CUDA_VISIBLE_DEVICES="5,6,7,8"  python -m torch.distributed.launch --nproc_per_node 4 ddp_rgcn.py


import sys
from time import time
from pathlib import Path
import logging

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0]  # CogKGE root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKGE root directory to PATH

import os
import torch
import torch.distributed as dist
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from cogkge import *
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def demo_basic(local_world_size, local_rank):
    init_seed(1+local_rank)
    # init_seed(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:0")

    # put your own code from here
    loader = FB15KLoader(dataset_path="../../dataset", download=True)
    train_data, valid_data, test_data = loader.load_all_data()
    node_lut, relation_lut = loader.load_all_lut()
    # loader.describe()
    # train_data.describe()
    # node_lut.describe()

    processor = FB15KProcessor(node_lut, relation_lut, reprocess=True)
    train_dataset = processor.process(train_data)
    valid_dataset = processor.process(valid_data)
    test_dataset = processor.process(test_data)
    node_lut, relation_lut = processor.process_lut()
    # node_lut.print_table(front=3)
    # relation_lut.print_table(front=3)

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)
    test_sampler = DistributedSampler(test_dataset)

    model = ComplEx(entity_dict_len=len(node_lut),
                    relation_dict_len=len(relation_lut),
                    embedding_dim=50,
                    penalty_weight=0.1)

    loss = NegLogLikehoodLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

    metric = Link_Prediction(link_prediction_raw=True,
                             link_prediction_filt=False,
                             batch_size=50000,
                             reverse=True)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,
        factor=0.5, min_lr=1e-9, verbose=True
    )

    negative_sampler = UnifNegativeSampler(triples=train_dataset,
                                           entity_dict_len=len(node_lut),
                                           relation_dict_len=len(relation_lut))
    trainer = Trainer(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_sampler=train_sampler,
        valid_sampler=valid_sampler,
        test_sampler=test_sampler,
        model=model,
        loss=loss,
        optimizer=optimizer,
        negative_sampler=negative_sampler,
        device=device,
        output_path="../../dataset",
        lookuptable_E=node_lut,
        lookuptable_R=relation_lut,
        metric=metric,
        trainer_batch_size=256,
        total_epoch=1000,
        lr_scheduler=lr_scheduler,
        apex=True,
        dataloaderX=True,
        num_workers=4,
        pin_memory=True,
        use_tensorboard_epoch=100,
        use_matplotlib_epoch=100,
        use_savemodel_epoch=100,
        use_metric_epoch=50,
        rank=local_rank,
    )
    dist.barrier()
    trainer.train()





def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    # env_dict = {
    #     key: os.environ[key]
    #     for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    # }
    # print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    # print(
    #     f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
    #     + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    # )

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--local_world_size", type=int, default=1)
    # args = parser.parse_args()
    spmd_main(local_world_size, local_rank)
