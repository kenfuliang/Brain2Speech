"""E022 raw-only technical gate and independent-process stochastic replay."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from causal_recipe_control import fresh_model,DayStream,objective,finite_state,smooth_tensor,INITIAL,PublishedGRU,KerasAdam
from run_causal_recipe_control import load,processed
from run_preprocessing_comparison import save_state,device
from research.audit import save_json,digest


def hashed(x):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):h.update(str((v.shape,v.dtype)).encode());h.update(v.cpu().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for t in v:visit(t)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()


def update(model,opt,sampler,dev):
    model.train();day,batch=sampler.batch(64)
    for g in opt.param_groups:g['lr']=.02*(1-g['iterations']/10000)
    opt.zero_grad(set_to_none=True);loss,pred=objective(model,batch,dev,day)
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step();finite_state(model,opt)
    return dict(loss=float(loss.detach().cpu()),prediction=pred,model=hashed(model.state_dict()),optimizer=hashed(opt.state_dict()),
        sampler=hashed(sampler.state()),torch_rng=hashed(torch.get_rng_state()),mps_rng=hashed(torch.mps.get_rng_state()),ids=[t['id']for t in batch])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args();run=a.run.resolve();dev=device()
    if a.child:
        s=torch.load(run/'gate-state.pt',map_location='cpu',weights_only=False)
        model=PublishedGRU().to(dev);model.load_state_dict(s['model']);opt=KerasAdam(model.parameters());opt.load_state_dict(s['optimizer'])
        sampler=DayStream(s['groups']);sampler.restore(s['sampler']);torch.set_rng_state(s['torch_rng']);torch.mps.set_rng_state(s['mps_rng'])
        save_json(run/'gate-observed.json',update(model,opt,sampler,dev));return
    protocol,data=load(run);norm=json.loads((run/'raw-normalization.json').read_text());p=processed(data,norm)
    groups=[[t for t in p['train']if t['session']==d]for d in protocol['days']]
    model,opt=fresh_model(dev);initial_sha=digest(INITIAL);initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    assert hashed(model.state_dict())==hashed(initial['model'])
    sampler=DayStream(groups);results=[update(model,opt,sampler,dev)for _ in range(2)]
    model.eval();x=torch.from_numpy(p['train'][0]['z'][None]).to(dev)
    with torch.inference_mode():
        y=model(smooth_tensor(x)).cpu();future=x.clone();future[:,64:]+=100
        changed=model(smooth_tensor(future)).cpu();prefix=model(smooth_tensor(x[:,:64])).cpu()
        torch.testing.assert_close(y[:,:8],changed[:,:8],rtol=0,atol=1e-6)
        torch.testing.assert_close(y[:,:8],prefix[:,:8],rtol=0,atol=1e-6)
        model(smooth_tensor(torch.flip(x,[1])));torch.testing.assert_close(y,model(smooth_tensor(x)).cpu(),rtol=0,atol=0)
    # Bound child serialization to the next batch, but preserve sampler sequence.
    sampler_state=sampler.state();next_day,next_batch=sampler.batch(64);sampler.restore(sampler_state)
    minimal_groups=[[dict(t,id=t['id'])for t in g]for g in groups]
    # Only z/labels/id are needed; drop raw copies to keep the gate file bounded.
    minimal_groups=[[{k:t[k]for k in ('id','z','labels')}for t in g]for g in minimal_groups]
    save_state(run/'gate-state.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),groups=minimal_groups,sampler=sampler.state(),
        torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()))
    expected=update(model,opt,sampler,dev);save_json(run/'gate-expected.json',expected)
    del model,opt,data,p,groups,minimal_groups,initial;gc.collect();torch.mps.empty_cache()
    subprocess.run([sys.executable,__file__,'--run',str(run),'--child'],check=True)
    observed=json.loads((run/'gate-observed.json').read_text());assert observed==expected
    assert digest(INITIAL)==initial_sha
    # Small-adapter mechanism is already covered by identity/gradient unit tests.
    save_json(run/'pilot.json',dict(status='pass',backend='mps',raw_only=True,technical_updates=3,
        batch=64,CTC_units='Unnormalized mean sentence loss',first_losses=[r['prediction']for r in results],
        future_prefix_tolerance=1e-6,trial_reset='exact',initialization='exact zero-update E017 ancestor',
        no_transformed_training=True,performance='Not measured by this gate',implementation_sha256=digest(__file__)))
    save_json(run/'resume-gate.json',dict(status='pass',fresh_process=True,
        exact=['Next batch IDs','loss','model','global KerasAdam slots','sampler state','CPU RNG','MPS RNG'],
        initial_unchanged=True,gate_state_sha256=digest(run/'gate-state.pt'),implementation_sha256=digest(__file__)))
    print('Raw technical gate and fresh-process replay passed',flush=True)


if __name__=='__main__':main()
