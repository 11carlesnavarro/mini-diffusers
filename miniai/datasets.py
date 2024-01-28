
from __future__ import annotations
import math,numpy as np, matplotlib.pyplot as plt
from operator import itemgetter
from itertools import zip_longest
import fastcore.all as fc

from torch.utils.data import default_collate
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

#import random
#from datasets import load_dataset,load_dataset_builder
import multiprocessing as mp

__all__ = ['inplace', 'collate_dict', 'DataLoaders', 'get_dls']

def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f

@inplace
def transformi(b):
    b[x] = [torch.flatten(TF.to_tensor(o)) for o in b[x]]

def collate_dict(ds):
    get = itemgetter(*ds.features)
    def _f(b): 
        return get(default_collate(b))
    return _f

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))

class DataLoaders:
    def __init__(self, *dls): 
        self.train,self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        f = collate_dict(dd['train'])
        return cls(*get_dls(*dd.values(), bs=batch_size, collate_fn=f, **kwargs))


if __name__ == '__main__':
    import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
    from pathlib import Path
    from torch import tensor,nn
    import torch.nn.functional as F

    name = "fashion_mnist"
    ds_builder = load_dataset_builder(name)
    dsd = load_dataset(name)

    train,test = dsd['train'],dsd['test']
    x,y = 'image','label'


    tsd = train.with_transform(transformi)
    dlf = DataLoader(tsd, batch_size=4, collate_fn=collate_dict(tsd))
    xb,yb = next(iter(dlf))
    print(xb.shape,yb)
