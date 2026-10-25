"""Two original-size stochastic GPU updates, not a full training run."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from pytorch_baseline import PublishedGRU,KerasAdam

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if shutil.disk_usage(a.output.parent).free<30*1024**3: raise RuntimeError('Storage guard')
    a.output.mkdir()
    torch.manual_seed(17)
    f=np.load(a.fixture)
    m=PublishedGRU(); m.load_tf(f); m.to('mps').train()
    opt=KerasAdam(m.parameters())
    raw=torch.tensor(f['raw'],device='mps')
    assert raw.shape[0]==64
    t=torch.arange(-4,5,device='mps',dtype=torch.float32)
    kernel=torch.exp(-t.square()/8); kernel/=kernel.sum()
    kernel=kernel.reshape(1,1,9).expand(256,1,9).contiguous()
    targets=torch.tensor(f['labels'].astype('int64'))-1
    lengths=torch.tensor(((f['lengths']-32)//4+1).astype('int64'))
    labels=torch.tensor(f['label_lengths'].astype('int64'))
    rows=[]
    for step in range(2):
        if shutil.disk_usage(a.output).free<30*1024**3: raise RuntimeError('Storage guard')
        start=time.monotonic()
        opt.zero_grad(set_to_none=True)
        x=raw+torch.randn_like(raw)+.2*torch.randn_like(raw[:,:1])
        x=torch.nn.functional.conv1d(x.transpose(1,2),kernel,padding=4,groups=256).transpose(1,2)
        y=m(x)
        loss=torch.nn.functional.ctc_loss(y.cpu().log_softmax(-1).transpose(0,1),targets,lengths,labels,blank=40,reduction='none').mean().to('mps')+m.regularization()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),10)
        assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
        opt.param_groups[0]['lr']=.02*(1-step/10000)
        opt.step(); torch.mps.synchronize()
        rows.append(dict(step=step,loss=loss.item(),gradient_norm=norm.item(),seconds=time.monotonic()-start,
                         allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory()))
        print(json.dumps(rows[-1]),flush=True)
        (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    m.eval()
    with torch.no_grad(): before=m(torch.tensor(f['input'][:2],device='mps')).cpu()
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=2,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),a.output/'checkpoint.pt')
    saved=torch.load(a.output/'checkpoint.pt',map_location='mps',weights_only=False)
    with torch.no_grad(): m.h0.add_(1)
    m.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
    with torch.no_grad(): after=m(torch.tensor(f['input'][:2],device='mps')).cpu()
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    result=dict(status='passed-two-step-gpu-pilot',batches=rows,batch_size=64,checkpoint_reload_exact=True,
                full_training_started=False,wer=None,ctc_device='cpu',neural_device='mps',
                limitations=['Two updates on the same first-day training batch, not independent convergence evidence','Stochastic cross-framework parity unverified','Concurrent host workloads; no controlled speedup estimate'])
    (a.output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
