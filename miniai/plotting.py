import fastcore.all as fc
import math,numpy as np, matplotlib.pyplot as plt
from itertools import zip_longest
import torch

__all__ = ['show_image', 'subplots', 'get_grid', 'show_images', 'plot_func']

@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ('data','cpu','permute')):
        im = im.data.cpu()
        if len(im.shape) == 3 and im.shape[0]<5:
            im = im.permute(1,2,0)
    elif not isinstance(im, np.ndarray):
        im = np.array(im)
    if im.shape[-1] == 1:
        im=im[...,0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks
    ax.set_yticks([])
    if noframe:
        ax.axis('off')
    return ax

@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows: int=1,
    ncols: int=1,
    figsize:tuple=None,
    imsize: int=3,
    suptitle:str=None,
    **kwargs
):
    "A figure and a set of subplots to display images of `imsize` inches"
    if figsize is None:
        figsize = (ncols*imsize, nrows*imsize)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows*ncols == 1:
        ax = np.array([ax])
    return fig,ax

fc.delegates(subplots)
def get_grid(
        n:int, 
        nrows:int=None,
        ncols:int=None,
        title:str=None,
        weight:str='bold',
        size:int=14,
        **kwargs,
):
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows: 
        ncols = ncols or int(np.floor(n/nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n/ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n/nrows))

    fig,axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows*ncols):
        axs.flat[i].set_axis_off()

    if title is not None:
        fig.suptitle(title, weight=weight, size=size)
    return fig, axs


@fc.delegates(subplots)
def show_images(
    ims: list,
    nrows:int|None=None,
    ncols:int|None=None,
    titles: list|None=None,
    **kwargs):

    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im, t, ax in zip_longest(ims, titles or [], axs):
        show_image(im, ax=ax, title=t)

def plot_func(f, start=-5., end=5., steps=100):
    x = torch.linspace(start, end, steps)
    plt.plot(x, f(x))
    plt.grid(True, which='both', ls='--')
    plt.axhline(y=0, color='k', linewidth=0.7)
    plt.axvline(x=0, color='k', linewidth=0.7)