"""E022: E017-compatible components with explicit causal/chronological departures."""
from pathlib import Path
import sys
import copy
import numpy as np
from scipy.signal import lfilter
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pytorch_baseline import PublishedGRU,KerasAdam
from preprocessing_comparison import split_rows
from research.data import load_trials

INITIAL=ROOT/'reproduction/results/retraining/pytorch-retrain-s17-20261007-01/initial.pt'
KINDS=('raw','sqrt','anscombe')
KERNEL=np.exp(-.5*(np.arange(9)/2)**2);KERNEL/=KERNEL.sum()


def transform(raw,kind):
    if kind not in KINDS:raise ValueError(kind)
    x=np.asarray(raw,dtype=np.float64).copy()
    if not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid raw observations; no cleaning')
    if kind=='sqrt':x[:,:128]=np.sqrt(x[:,:128])
    if kind=='anscombe':x[:,:128]=2*np.sqrt(x[:,:128]+.375)
    # E017 uses spike power directly, not log1p(power).
    return x


def moments(trials,kind):
    total=np.zeros(256);second=np.zeros(256);n=0
    for t in trials:
        x=transform(t['raw'],kind);n+=len(x);total+=x.sum(0);second+=(x*x).sum(0)
    if not n:raise ValueError('No permitted normalization observations')
    mu=total/n;variance=np.maximum(second/n-mu*mu,0)
    return dict(mean=mu.tolist(),std=np.sqrt(variance).tolist(),bins=n,ids=[t['id']for t in trials])


def fit_normalizers(trials,kind):
    global_stats=moments(trials,kind)
    per_day={d:moments([t for t in trials if t['session']==d],kind)for d in sorted({t['session']for t in trials})}
    return dict(kind=kind,global_stats=global_stats,per_day=per_day)


def standardize(raw,kind,stats,fallback=None):
    sd=np.array(stats['std']);mu=np.array(stats['mean'])
    if fallback is not None:sd=np.where(sd<=1e-8,np.array(fallback['std']),sd)
    return ((transform(raw,kind)-mu)/(sd+1e-8)).astype(np.float32)


def smooth(x):return lfilter(KERNEL,[1.],x,axis=0).astype(np.float32)


def smooth_tensor(x):
    kernel=torch.tensor(KERNEL[::-1].copy(),dtype=x.dtype,device=x.device).reshape(1,1,9).expand(256,1,9)
    return F.conv1d(F.pad(x.transpose(1,2),(8,0)),kernel,groups=256).transpose(1,2)


class DayStream:
    """Whole-session shuffle then repeat before batching; all sessions <1024 trials.

    NumPy RNG differs from TensorFlow; permutation/repeat semantics are explicit.
    """
    def __init__(self,groups,seed=17):
        assert all(len(g)<=1024 for g in groups)
        self.groups=groups;self.rng=np.random.RandomState(seed)
        self.order=[self.rng.permutation(len(g)).tolist()for g in groups];self.positions=[0]*len(groups)

    def batch(self,size=64):
        day=int(self.rng.choice(len(self.groups)));indices=[]
        while len(indices)<size:
            if self.positions[day]==len(self.order[day]):
                self.order[day]=self.rng.permutation(len(self.groups[day])).tolist();self.positions[day]=0
            k=min(size-len(indices),len(self.order[day])-self.positions[day]);start=self.positions[day]
            indices.extend(self.order[day][start:start+k]);self.positions[day]+=k
        return day,[self.groups[day][i]for i in indices]

    def state(self):return copy.deepcopy(dict(rng=self.rng.get_state(),order=self.order,positions=self.positions))
    def restore(self,s):self.rng.set_state(s['rng']);self.order=copy.deepcopy(s['order']);self.positions=list(s['positions'])


def native_batch(trials,device,augment=False):
    lengths=[len(t['z'])for t in trials];x=np.zeros((len(trials),max(lengths),256),np.float32);frames=[]
    for i,t in enumerate(trials):
        x[i,:lengths[i]]=t['z'];y=t['labels'];n=(lengths[i]-32)//4+1
        if n<len(y)+int(np.sum(y[1:]==y[:-1])):raise ValueError('Impossible CTC '+t['id'])
        frames.append(n)
    x=torch.from_numpy(x).to(device)
    if augment:x=x+torch.randn_like(x)+.2*torch.randn_like(x[:,:1])
    x=smooth_tensor(x)
    return x,torch.from_numpy(np.concatenate([t['labels']for t in trials])),torch.tensor(frames),torch.tensor([len(t['labels'])for t in trials])


def objective(model,trials,device,day,augment=True,regularization=True):
    x,labels,frames,k=native_batch(trials,device,augment)
    logits=model(x,day).cpu().log_softmax(-1).transpose(0,1)
    prediction=F.ctc_loss(logits,labels,frames,k,blank=40,reduction='none',zero_infinity=False).mean()
    loss=prediction.to(device)+(model.regularization(day)if regularization else 0)
    return loss,float(prediction.detach())


def fresh_model(device):
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    assert initial['next_attempt']==0 and not initial['history'] and not initial['optimizer']['state']
    model=PublishedGRU().to(device);model.load_state_dict(initial['model'])
    optimizer=KerasAdam(model.parameters(),lr=.02,eps=.1)
    optimizer.load_state_dict(initial['optimizer'])
    torch.set_rng_state(initial['torch_rng'])
    if device.type=='mps':torch.mps.set_rng_state(initial['mps_rng'])
    return model,optimizer


def finite_state(model,optimizer):
    for p in model.parameters():
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite parameter')
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
    for state in optimizer.state.values():
        for name in ('m','v'):
            if name in state and not torch.isfinite(state[name]).all():raise FloatingPointError('Nonfinite Adam '+name)


class FastInput(nn.Module):
    def __init__(self):
        super().__init__();self.log_gain=nn.Parameter(torch.zeros(256));self.offset=nn.Parameter(torch.zeros(256))
        generator=torch.Generator().manual_seed(17)
        self.u=nn.Parameter(torch.randn(256,8,generator=generator)/16);self.v=nn.Parameter(torch.zeros(8,256))
    def forward(self,z):
        x=z*self.log_gain.clamp(-.7,.7).exp()+self.offset.clamp(-2,2)
        return x+(x@self.v.T)@self.u.T
    def regularization(self):return self.log_gain.square().mean()+self.offset.square().mean()+(self.u@self.v).square().mean()


class TargetModel(nn.Module):
    def __init__(self,parent,reference_day=13):
        super().__init__();self.parent=parent;self.fast=FastInput().to(next(parent.parameters()).device);self.reference_day=reference_day
        self.parent.requires_grad_(False);self.eval()
    def forward(self,x,day=None):return self.parent(self.fast(x),self.reference_day)
