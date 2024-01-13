
__all__ = ['def_device', 'conv', 'to_device', 'collate_device']

import torch

from torch.utils.data import default_collate
from typing import Mapping

from .training import *
from .datasets import *

def conv(ni, nf, ks=3, stride=2, act=True):
    res = torch.nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)
    if act:
        res = torch.nn.Sequential(res, torch.nn.ReLU())
    return res

def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k:to_device(v, device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)

def collate_device(b):
    return to_device(default_collate(b))