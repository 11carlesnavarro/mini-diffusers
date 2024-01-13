import torch.nn.functional as F
import torch
from torch import optim
from miniai.training import *
from miniai.datasets import *

from torch.utils.data import default_collate
from typing import Mapping
from pathlib import Path
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl, numpy as np


path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'
with gzip.open(path_gz, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

x_train, y_train, x_valid, y_valid = map(torch.tensor, [x_train, y_train, x_valid, y_valid])

def conv(ni, nf, ks=3, stride=2, act=True):
    res = torch.nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2)
    if act:
        res = torch.nn.Sequential(res, torch.nn.ReLU())
    return res

simple_cnn = torch.nn.Sequential(
    conv(1,4),
    conv(4, 8),
    conv(8, 16),
    conv(16, 16),
    conv(16, 10, act=False),
    torch.nn.Flatten(),
)

x_imgs = x_train.view(-1,1,28,28)
xv_imgs = x_valid.view(-1,1,28,28)
train_ds,valid_ds = Dataset(x_imgs, y_train),Dataset(xv_imgs, y_valid)

#|export
def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)

def collate_device(b): 
    return to_device(default_collate(b))

bs = 256
lr = 0.4
train_dl, valid_dl = get_dls(train_ds, valid_ds, bs, collate_fn = collate_device)
opt = optim.SGD(simple_cnn.parameters(), lr=lr)

loss, acc = fit(5, simple_cnn.to(def_device), F.cross_entropy, opt, train_dl, valid_dl)