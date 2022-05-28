import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
# from ordered_set import OrderedSet

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
# from torch_scatter import scatter_add
from .util_scatter import scatter_add

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2


    def rfft(x, d):
        t = rfft2(x, dim=(-d))
        return torch.stack((t.real, t.imag), -1)


    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d))

np.set_printoptions(precision=4)


def set_gpu(gpus):
    """
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------
		
	"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return irfft(com_mult(rfft(a, 1), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def construct_adj(train_dataset, relation_dict_len):
    edge_index, edge_type = [], []
    if train_dataset.data.shape[1] == 3: # score_based
        for sub, rel, obj in train_dataset.data:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        for sub, rel, obj in train_dataset.data:
            edge_index.append((obj, sub))
            edge_type.append(rel + relation_dict_len)
    else:  # classification-based
        label = train_dataset.label_data
        for j,(sub, rel) in enumerate(train_dataset.data):
            for elem in torch.nonzero(label[j]):
                e2_idx = elem.item()
                edge_index.append((sub,e2_idx))
                edge_type.append(rel)

        for j,(sub, rel) in enumerate(train_dataset.data):
            for elem in torch.nonzero(label[j]):
                e2_idx = elem.item()
                edge_index.append((e2_idx,sub))
                edge_type.append(rel + relation_dict_len)

    return edge_index,edge_type