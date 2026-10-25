"""E021 matched count transforms and restricted adapter; no import-time jobs."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import lfilter
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'reproduction'))
from pytorch_baseline import PublishedGRU
from research.data import PHONES, labels

TRANSFORMS=('raw','sqrt','anscombe')
KERNEL=np.exp(-.5*(np.arange(9)/2)**2);KERNEL/=KERNEL.sum()


def transform(raw,kind):
    if kind not in TRANSFORMS:raise ValueError(kind)
    raw=np.asarray(raw,dtype=np.float64)
    if not np.isfinite(raw).all() or (raw<0).any():raise ValueError('Invalid features; no cleaning')
    x=raw.copy()
    if kind=='sqrt':x[:,:128]=np.sqrt(x[:,:128])
    if kind=='anscombe':x[:,:128]=2*np.sqrt(x[:,:128]+.375)
    x[:,128:]=np.log1p(x[:,128:])
    return x


def fit_stats(trials,kind):
    # Equal trial weight; fitting is restricted to the initial training subset.
    mean=np.zeros(256);second=np.zeros(256)
    for trial in trials:
        x=transform(trial['raw'],kind);mean+=x.mean(0);second+=(x*x).mean(0)
    mean/=len(trials);second/=len(trials)
    return dict(kind=kind,mean=mean.tolist(),variance=np.maximum(second-mean*mean,1e-4).tolist(),
                trial_ids=[t['id'] for t in trials],weighting='equal trial')


def preprocess(raw,stats):
    z=(transform(raw,stats['kind'])-np.array(stats['mean']))/np.sqrt(np.array(stats['variance']))
    return lfilter(KERNEL,[1.],z,axis=0).astype(np.float32)


class SmallAdapter(nn.Module):
    def __init__(self,rank=8):
        super().__init__();self.shared=nn.Linear(256,256)
        nn.init.eye_(self.shared.weight);nn.init.zeros_(self.shared.bias)
        self.log_gain=nn.Parameter(torch.zeros(256));self.offset=nn.Parameter(torch.zeros(256))
        self.u=nn.Parameter(torch.randn(256,rank)/16)
        self.v=nn.Parameter(torch.zeros(rank,256))
        self.adaptation(False)

    @property
    def weight(self):return self.shared.weight

    def adaptation(self,enabled=True):
        for p in (self.log_gain,self.offset,self.u,self.v):p.requires_grad_(enabled)

    def forward(self,z):
        x=self.log_gain.clamp(-.7,.7).exp()*z+self.offset.clamp(-2.,2.)
        return self.shared(x)+(x@self.v.T)@self.u.T

    def regularization(self):
        return self.log_gain.square().mean()+self.offset.square().mean()+(self.u@self.v).square().mean()


class ComparisonGRU(PublishedGRU):
    def __init__(self):
        super().__init__()
        # Only one shared mapping, not supervised target-day adapter lookup.
        self.adapters=nn.ModuleList([SmallAdapter()])

    def calibrate_only(self):
        for p in self.parameters():p.requires_grad_(False)
        self.adapters[0].adaptation(True)
        self.eval() # Frozen backbone dropout disabled, autograd still enabled.


def split_rows(manifest):
    train=[];selection=[];calibration={};evaluation={}
    for day in manifest['folds']['initial_train']:
        rows=manifest['sessions'][day]['train']['trials']
        blocks=sorted({r['block'] for r in rows})
        if len(blocks)<2:raise ValueError('Insufficient chronological training blocks')
        train.extend(dict(r,session=day,partition='train') for r in rows if r['block']!=blocks[-1])
        selection.extend(dict(r,session=day,partition='train') for r in rows if r['block']==blocks[-1])
    for day in manifest['folds']['development']:
        calibration[day]=[dict(r,session=day,partition='train') for r in manifest['sessions'][day]['train']['trials']]
        evaluation[day]=[dict(r,session=day,partition='test') for r in manifest['sessions'][day]['test']['trials']]
        if max(r['block'] for r in calibration[day])>=min(r['block'] for r in evaluation[day]):
            raise ValueError('Calibration is not earlier than evaluation')
    forbidden=set(manifest['folds']['final'])
    all_rows=train+selection+sum(calibration.values(),[])+sum(evaluation.values(),[])
    assert not forbidden&{r['session'] for r in all_rows}
    assert len({r['id'] for r in all_rows})==len(all_rows)
    return dict(train=train,selection=selection,calibration=calibration,evaluation=evaluation)


def tensors(trials,device):
    sizes=[len(t['x']) for t in trials];x=np.zeros((len(trials),max(sizes),256),np.float32)
    lengths=[]
    for i,t in enumerate(trials):
        x[i,:sizes[i]]=t['x'];n=(sizes[i]-32)//4+1
        target=t['labels'];minimum=len(target)+int(np.sum(target[1:]==target[:-1]))
        if n<minimum:raise ValueError('Impossible CTC path '+t['id'])
        lengths.append(n)
    return (torch.from_numpy(x).to(device),torch.from_numpy(np.concatenate([t['labels'] for t in trials])),
            torch.tensor(lengths),torch.tensor([len(t['labels']) for t in trials]))


def ctc(model,trials,device):
    x,y,n,k=tensors(trials,device)
    logits=model(x).cpu().log_softmax(-1).transpose(0,1)
    return torch.nn.functional.ctc_loss(logits,y,n,k,blank=40,zero_infinity=False)
