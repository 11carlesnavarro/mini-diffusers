import torch
from typing import Mapping
from torch.utils.data import default_collate
import random
import numpy as np

def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor): 
        return x.to(device)
    if isinstance(x, Mapping): 
        return {k:v.to(device) for k,v in x.items()}
    
    return type(x)(to_device(o, device) for o in x)

def to_cpu(x):
    if isinstance(x, Mapping):
        return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x,list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res

def collate_device(b): 
    return to_device(default_collate(b))

def set_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals():
        return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop('_i' + repr(n), None)
    user_ns.update(dict(_i='',__ii='', _iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ''


def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'):
        delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'):
        delattr(sys, 'last_value')

def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda_empty_cache()