import math
import torch
from torch.utils.data import default_collate, DataLoader
from fastprogress import progress_bar

def alpha_bar(t):
    return (t * math.pi / 2).cos()**2

def inv_alpha_bar(x):
    return x.sqrt().acos() * 2 / math.pi

def noisify(x0):
    device = x0.device
    n = len(x0)
    t = torch.rand(n, device=device).to(x0).clamp(0, 0.999)
    eps = torch.randn(x0.shape, device=device)
    alpha_bar_t = alpha_bar(t).reshape(-1, 1, 1, 1).to(device)
    xt = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * eps
    return (xt, t), eps

def collate_ddpm(b):
    return noisify(default_collate(b)['image']) # weird

def dl_ddpm(ds, bs=1, num_workers=4): 
    return DataLoader(ds, batch_size=bs, collate_fn=collate_ddpm, num_workers=num_workers)

def ddim_step(x_t, noise, alpha_bar_t, alpha_bar_t1, beta_bar_t, beta_bar_t1, eta, sig, clamp=True):
    sig = ((beta_bar_t1 / beta_bar_t).sqrt() * (1 - alpha_bar_t/alpha_bar_t1).sqrt()) * eta
    x_0_hat = ((x_t - (1 - alpha_bar_t).sqrt() * noise) / alpha_bar_t.sqrt())
    if clamp:
        x_0_hat = x_0_hat.clamp(-1, 1)
    if beta_bar_t1 <= sig ** 2 + 0.01:
        sig = 0.
    x_t = alpha_bar_t1.sqrt() * x_0_hat + (beta_bar_t1 -  sig ** 2).sqrt() * noise
    x_t += sig * torch.randn(x_t.shape).to(x_t)
    return x_0_hat, x_t

@torch.no_grad()
def sample(f, model, sz, steps, eta=1., clamp=True):
    model.eval()
    ts = torch.linspace(1-1/steps, 0, steps)
    x_t = torch.randn(sz).cuda()
    preds = []

    for i, t in enumerate(progress_bar(ts)):
        t = t[None].cuda()
        alpha_bar_t = alpha_bar(t)
        noise = model((x_t, t))
        alpha_bar_t1 = alpha_bar(t - 1/steps) if t >=1/steps else torch.tensor(1)
        x_0_hat, x_t = f(x_t, noise, alpha_bar_t, alpha_bar_t1, 1-alpha_bar_t, 1-alpha_bar_t1, eta, 1 - ((i + 1) / 100), clamp=clamp)
        preds.append(x_0_hat.float().cpu())
    return preds


