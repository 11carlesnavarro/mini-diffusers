from functools import partial
from miniai.init import GeneralRelu
from miniai.conv import conv
import fastcore.all as fc
import torch.nn as nn

__all__ = ['act_gr', 'ResBlock']


act_gr = partial(GeneralRelu, leak=0.1, sub=0.4)

def _conv_block(ni, nf, stride, act=act_gr, norm=None, ks=3):
    return nn.Sequential(
        conv(ni, nf, ks, stride, act=act, norm=norm, ks=ks),
        conv(nf, nf, ks, 1, act=None, norm=norm, ks=ks)
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, act=act_gr, norm=None):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride, act=act, norm=norm)
        self.idconv = fc.noop if ni==nf else conv(ni, nf, ks=1, stride=1, act=NoneB)
        self.pool = fc.noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
        self.act = act()

    def forward(self, x):
        return act_gr(self.convs(x) + self.idconv(self.pool(x)))