import os
import sys
import tempfile
import torch
import torch.distributed as dist
# import torch.distributed.launch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29057'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
class ToyModel(nn.Module):
    #  *.asdf
    # //oh!!!!!!!!!!1
    # !红色
    # todo 芥末
    # todo test
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(local_world_size, local_rank):

    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    n = 1
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )

    model = ToyModel().cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    # print("Cuda visible devices count:",torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    spmd_main(args.local_world_size, args.local_rank)
    a = 1
    b = 2
    a = a + b
 


# import os
# import sys
# import tempfile
# import torch
# import torch.distributed as dist
# # import torch.distributed.launch
# import torch.nn as nn
# import torch.optim as optim
# import torch.multiprocessing as mp
# import argparse

# from torch.nn.parallel import DistributedDataParallel as DDP

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()
# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def demo_basic(rank, world_size):
#     print(f"Running basic DDP example on rank {rank}.")
#     setup(rank, world_size)

#     # create model and move it to GPU with id rank
#     model = ToyModel().to(rank)
#     ddp_model = DDP(model, device_ids=[rank])

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(rank)
#     loss_fn(outputs, labels).backward()
#     optimizer.step()

#     cleanup()


# def run_demo(demo_fn, world_size):
#     mp.spawn(demo_fn,
#              args=(world_size,),
#              nprocs=world_size,
#              join=True)

# def demo_checkpoint(rank, world_size):
#     print(f"Running DDP checkpoint example on rank {rank}.")
#     setup(rank, world_size)

#     model = ToyModel().to(rank)
#     ddp_model = DDP(model, device_ids=[rank])

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
#     if rank == 0:
#         # All processes should see same parameters as they all start from same
#         # random parameters and gradients are synchronized in backward passes.
#         # Therefore, saving it in one process is sufficient.
#         torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

#     # Use a barrier() to make sure that process 1 loads the model after process
#     # 0 saves it.
#     dist.barrier()
#     # configure map_location properly
#     map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#     ddp_model.load_state_dict(
#         torch.load(CHECKPOINT_PATH, map_location=map_location))

#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(rank)
#     loss_fn = nn.MSELoss()
#     loss_fn(outputs, labels).backward()
#     optimizer.step()

#     # Not necessary to use a dist.barrier() to guard the file deletion below
#     # as the AllReduce ops in the backward pass of DDP already served as
#     # a synchronization.

#     if rank == 0:
#         os.remove(CHECKPOINT_PATH)

#     cleanup()

# class ToyMpModel(nn.Module):
#     def __init__(self, dev0, dev1):
#         super(ToyMpModel, self).__init__()
#         self.dev0 = dev0
#         self.dev1 = dev1
#         self.net1 = torch.nn.Linear(10, 10).to(dev0)
#         self.relu = torch.nn.ReLU()
#         self.net2 = torch.nn.Linear(10, 5).to(dev1)

#     def forward(self, x):
#         x = x.to(self.dev0)
#         x = self.relu(self.net1(x))
#         x = x.to(self.dev1)
#         return self.net2(x)
# def demo_model_parallel(rank, world_size):
#     print(f"Running DDP with model parallel example on rank {rank}.")
#     setup(rank, world_size)

#     # setup mp_model and devices for this process
#     dev0 = (rank * 2) % world_size
#     dev1 = (rank * 2 + 1) % world_size
#     mp_model = ToyMpModel(dev0, dev1)
#     ddp_mp_model = DDP(mp_model)

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

#     optimizer.zero_grad()
#     # outputs will be on dev1
#     outputs = ddp_mp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(dev1)
#     loss_fn(outputs, labels).backward()
#     optimizer.step()

#     cleanup()
# def spmd_main(local_world_size, local_rank):
#     # These are the parameters used to initialize the process group
#     env_dict = {
#         key: os.environ[key]
#         for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
#     }
#     print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
#     dist.init_process_group(backend="nccl")
#     print(
#         f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
#         + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
#     )

#     demo_basic(local_world_size, local_rank)

#     # Tear down the process group
#     dist.destroy_process_group()
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument("--local_world_size", type=int, default=1)
#     args = parser.parse_args()
#     spmd_main(args.local_world_size, args.local_rank)


# if __name__ == "__main__":
#     n_gpus = torch.cuda.device_count()
#     assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
#     world_size = n_gpus
#     run_demo(demo_basic, world_size)
    # run_demo(demo_checkpoint, world_size)
    # run_demo(demo_model_parallel, world_size)


# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP


# def example(rank, world_size):
#     # create default process group
#     # dist.init_process_group("gloo", rank=rank, world_size=world_size)
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     # create local model
#     model = nn.Linear(10, 10).to(rank)
#     # construct DDP model
#     ddp_model = DDP(model, device_ids=[rank])
#     # define loss function and optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     # forward pass
#     outputs = ddp_model(torch.randn(20, 10).to(rank))
#     labels = torch.randn(20, 10).to(rank)
#     # backward pass
#     loss_fn(outputs, labels).backward()
#     # update parameters
#     optimizer.step()

# def main():
#     world_size = 2
#     mp.spawn(example,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True)

# if __name__=="__main__":
#     main()


# import os
# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# RANK = int(os.getenv('RANK', -1))
# WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# print(LOCAL_RANK,RANK,WORLD_SIZE)
# ## main.py文件
# import torch
# import torch.nn as nn
# import torch.optim as optim
# # 新增：
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# # 新增：从外面得到local_rank参数
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", default=-1)
# args = parser.parse_args()
# local_rank = args.local_rank
# print("Local Rank:",local_rank)
# # print(local_rank)

# # 新增：DDP backend初始化
# torch.cuda.set_device(local_rank)
# dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# # 构造模型
# device = torch.device(f'cuda:{args.local_rank}')

# model = nn.Linear(10, 10).to(device)
# # 新增：构造DDP model
# # model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# # # 前向传播
# # outputs = model(torch.randn(20, 10).to(local_rank))
# # labels = torch.randn(20, 10).to(local_rank)
# # loss_fn = nn.MSELoss()
# # loss_fn(outputs, labels).backward()
# # # 后向传播
# # optimizer = optim.SGD(model.parameters(), lr=0.001)
# # optimizer.step()
