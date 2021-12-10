#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29051 ddp_test.py 

