"""Minimal E025 observation head; E022 backbone and CTC recipe unchanged."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from causal_recipe_control import native_batch


class CountHead(nn.Module):
    def __init__(self,means):
        super().__init__()
        # Head construction must not alter the paired backbone dropout/noise RNG.
        state=torch.get_rng_state()
        self.linear=nn.Linear(1024,512)
        with torch.no_grad():
            self.linear.weight.zero_()
            mean=torch.as_tensor(np.tile(np.maximum(means,1e-4),4),dtype=torch.float32)
            self.linear.bias.copy_(torch.log(torch.expm1(mean)))
        torch.set_rng_state(state)

    def forward(self,h):return F.softplus(self.linear(h)).reshape(*h.shape[:2],4,128)+1e-6


def count_targets(trials,frames,device):
    """Four nonoverlapping20ms targets per patch-end; skip first28warmup bins."""
    nmax=int(frames.max());targets=np.zeros((len(trials),nmax,4,128),np.float32)
    valid=np.zeros((len(trials),nmax),bool)
    for i,t in enumerate(trials):
        n=int(frames[i]);counts=np.asarray(t['raw'])[:,:128]
        if not np.isfinite(counts).all() or (counts<0).any() or not np.equal(counts,np.floor(counts)).all():
            raise ValueError('Poisson targets must be verified nonnegative integer counts')
        targets[i,:n]=counts[28:28+4*n].reshape(n,4,128);valid[i,:n]=True
    return torch.from_numpy(targets).to(device),torch.from_numpy(valid).to(device)


def reconstruction(rates,targets,valid):
    # Mean over channels/subbins, sum over frames, then equal-trial mean.
    # Includes exact lgamma constant for diagnostic count NLL.
    nll=rates-targets*torch.log(rates)+torch.lgamma(targets+1)
    frame=nll.mean((-1,-2));per_trial=(frame*valid).sum(1)
    return per_trial.mean(),(frame*valid).sum()/valid.sum()


def loss(model,head,trials,device,day,weight,augment=True):
    x,labels,frames,k=native_batch(trials,device,augment);capture=[]
    hook=model.layers[-1].register_forward_hook(lambda module,args,result:capture.append(result[0]))
    try:logits=model(x,day)
    finally:hook.remove()
    ctc=F.ctc_loss(logits.cpu().log_softmax(-1).transpose(0,1),labels,frames,k,blank=40,reduction='none',zero_infinity=False).mean()
    total=ctc.to(device)+model.regularization(day)
    if weight:
        target,valid=count_targets(trials,frames,device);rec,mean_nll=reconstruction(head(capture[0]),target,valid)
        total=total+weight*rec
    else:rec=mean_nll=torch.zeros((),device=device)
    return total,dict(CTC=float(ctc.detach()),reconstruction_sentence=float(rec.detach().cpu()),count_NLL=float(mean_nll.detach().cpu()))
