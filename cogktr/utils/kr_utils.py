import datetime
import os
import random

import numpy as np
import torch


# import the specified class
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


# compute the output path from the data path
def cal_output_path(data_path, model_name):
    output_path = os.path.join(*data_path.split("/")[:], "experimental_output",
                               model_name + str(datetime.datetime.now())[:-4]).replace(
        ':', '-').replace(' ', '--')
    return output_path


def init_cogktr(device_id, seed):
    """
    cogktr初始化

    :param device_id: 使用GPU:比如输入GPU编号 "0,1"或者“cuda:0,cuda:1”;使用CPU:输入“cpu”
    :param seed: 随机数种子
    :return: device
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device_list = str(device_id).strip().lower().replace('cuda:', '')
    cpu = device_list == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device_list:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device_list  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device_list} requested'  # check availability
    device = torch.device('cuda' if torch.cuda.is_available() == True else "cpu")
    return device

