import torch.nn.functional as F

__all__  = ['accuracy', 'Dataset', 'fit', 'get_dls']

import pickle,gzip,math,os,time,shutil,torch, matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
from pathlib import Path
from torch import tensor,nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler

def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()

def report(loss, preds, yb):
    print(f'loss: {loss:.2f}, accuracy: {accuracy(preds, yb):.2f}')

class Dataset():
    def __init__(self, x, y):
        self.x,self.y = x,y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i],self.y[i]

def fit(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss, tot_acc, count = 0.,0.,0
            for xb,yb in val_dl:
                pred = model(xb)
                n = len(xb)
                tot_loss += loss_func(pred, yb).item() #*n
                tot_acc  += accuracy(pred,yb).item() #*n
                count += 1
        print(epoch, tot_loss/count, tot_acc/count)
    return tot_loss/count, tot_acc/count

def get_dls(train_ds, val_ds, bs, **kwargs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=bs*2, **kwargs)
    )