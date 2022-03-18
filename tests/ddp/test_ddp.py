# command:python -m torch.distributed.launch --nproc_per_node 2 test_ddp.py


import sys
from pathlib import Path

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
    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].

    torch.cuda.set_device(local_rank)

    device = torch.device("cuda:0")
    print("Total device count:", torch.cuda.device_count(), " And current device:", torch.cuda.current_device())

    loader = FB15KLoader(dataset_path="../../dataset", download=True)
    train_data, valid_data, test_data = loader.load_all_data()
    node_lut, relation_lut = loader.load_all_lut()

    processor = FB15KProcessor(node_lut, relation_lut, reprocess=True, train_pattern="classification_based")
    train_dataset = processor.process(train_data)
    valid_dataset = processor.process(valid_data)
    test_dataset = processor.process(test_data)
    node_lut, relation_lut = processor.process_lut()

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)
    test_sampler = DistributedSampler(test_dataset)

    model = TuckER(entity_dict_len=len(node_lut),
                   relation_dict_len=len(relation_lut),
                   d1=200,
                   d2=200,
                   input_dropout=0.2,
                   hidden_dropout1=0.2,
                   hidden_dropout2=0.3)

    loss = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0)

    metric = Link_Prediction(link_prediction_raw=True,
                             link_prediction_filt=True,
                             batch_size=5000000,
                             reverse=False,
                             metric_pattern="classification_based")

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,
        factor=0.5, min_lr=1e-9, verbose=True
    )

    negative_sampler = UnifNegativeSampler(triples=train_dataset,
                                           entity_dict_len=len(node_lut),
                                           relation_dict_len=len(relation_lut),
                                           node_lut=node_lut)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler,
                              batch_size=1024, num_workers=1, shuffle=False)
    model.set_model_config(model_loss=loss,
                           model_metric=None,
                           model_negative_sampler=negative_sampler,
                           model_device=device, )
    model = model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False
                )
    for epoch in range(50):
        train_sampler.set_epoch(epoch)
        model.train()
        for train_step, batch in enumerate(tqdm(train_loader)):
            train_loss = model.module.loss(batch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


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
