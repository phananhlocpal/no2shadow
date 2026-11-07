"""Multi-station quantile forecaster used in the paper (called RGQF in the code).

Each station is encoded by a GRU over three channels (masked value, observation mask, normalised age). Stations exchange
information through attention whose scores are down-weighted by each station's share of missing hours in the last 24 h.
The head returns the 10%, 50% and 90% quantiles for every horizon. Missing inputs are zero in standardised space.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class RGQF(nn.Module):
    def __init__(self, hidden, horizon, gate=True):
        super().__init__()
        self.horizon = horizon
        self.gate = gate
        self.encoder = nn.GRU(3, hidden, batch_first=True)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.mix = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU())
        self.head = nn.Linear(hidden, horizon*3)
        self.log_quality_weight = nn.Parameter(torch.zeros(()))

    def forward(self, values, mask):
        b, length, c = values.shape
        pos = torch.arange(length, device=values.device).view(1,length,1).expand_as(mask)
        latest = torch.cummax(torch.where(mask.bool(), pos+1, torch.zeros_like(pos)), dim=1).values
        age = ((pos+1-latest).float()/length).clamp(0,1)
        features = torch.stack((values*mask, mask, age), dim=-1).permute(0,2,1,3).reshape(b*c,length,3)
        _, state = self.encoder(features)
        z = state[-1].reshape(b,c,-1)
        scores = torch.matmul(self.q(z),self.k(z).transpose(1,2)) / (z.shape[-1] ** .5)
        if self.gate:
            quality = mask[:, -min(24,length):, :].mean(dim=1)
            scores = scores + F.softplus(self.log_quality_weight) * torch.log(quality.clamp_min(1e-4))[:,None,:]
        a = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(a,self.v(z))
        raw = self.head(self.mix(torch.cat([z,mixed],dim=-1))).reshape(b,c,self.horizon,3).permute(0,2,1,3)
        median = raw[...,0]
        low = median-F.softplus(raw[...,1])
        high = median+F.softplus(raw[...,2])
        return torch.stack((low,median,high),dim=-1)


def pinball(prediction, target, observed):
    quantiles = prediction.new_tensor([.1,.5,.9])
    residual = target.unsqueeze(-1)-prediction
    value = torch.maximum(quantiles*residual,(quantiles-1)*residual)*observed.unsqueeze(-1)
    return value.sum() / (observed.sum().clamp_min(1)*3)


def augment_blocks(values, mask, probability, block):
    """Randomly hide one contiguous segment per selected series; no future data used."""
    b,l,c=mask.shape
    choose=(torch.rand((b,c),device=mask.device)<probability)
    start=torch.randint(0,max(1,l-block+1),(b,c),device=mask.device)
    t=torch.arange(l,device=mask.device)[None,:,None]
    hide=choose[:,None,:] & (t>=start[:,None,:]) & (t<start[:,None,:]+block)
    modified=mask.masked_fill(hide,0)
    return values*modified, modified


class Windows(Dataset):
    """Sliding (context, horizon) windows over a standardised panel; yields inputs, masks, targets, target masks, origin."""

    def __init__(self, values, mask, start, end, context, horizon, stride):
        self.x, self.m, self.context, self.horizon = values, mask, context, horizon
        self.origins = list(range(max(start, context), end - horizon + 1, stride))

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        t, l, h = self.origins[i], self.context, self.horizon
        return (torch.from_numpy(self.x[t - l:t]), torch.from_numpy(self.m[t - l:t]),
                torch.from_numpy(self.x[t:t + h]), torch.from_numpy(self.m[t:t + h]), t)
