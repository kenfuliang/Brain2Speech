"""Resumable E025 single auxiliary-loss arm; completed E022 raw is matched control."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from poisson_recipe import CountHead,loss
from causal_recipe_control import fresh_model,KerasAdam,DayStream,finite_state
from run_causal_recipe_control import ROOT,load,processed,selection
from run_preprocessing_comparison import save_state,device
from research.audit import digest,save_json,space_guard


def main(out):
    q=json.loads((out/'protocol.json').read_text());assert json.loads((out/'pilot.json').read_text())['status']=='pass'
    for path,h in q['code_hashes'].items():assert digest(path)==h
    for path,h in q['parent_source_hashes'].items():assert digest(path)==h
    import shutil
    gate=json.loads((out/'storage-gate.json').read_text());free=shutil.disk_usage(out).free
    if free<gate['required_GiB']*2**30:raise RuntimeError('Full-budget planning storage gate: require47.32GiBfree; do not delete artifacts or bypass')
    source=ROOT/'research/runs/e022-causal-recipe-s17-20261012-01'
    assert digest(source/'protocol.json')==q['parent_protocol_sha256']
    p,data=load(source);norm=json.loads((source/'raw-normalization.json').read_text());prepared=processed(data,norm)
    folder=out/'poisson';folder.mkdir(exist_ok=True)
    execution=dict(protocol_sha256=digest(out/'protocol.json'),runner_sha256=digest(__file__),weight=.1,budget=10001,seed=17)
    if (folder/'execution.json').exists():assert json.loads((folder/'execution.json').read_text())==execution
    else:save_json(folder/'execution.json',execution)
    if (folder/'training-complete.json').exists():return
    dev=device();model,_=fresh_model(dev);head=CountHead(np.array(norm['global_stats']['mean'])[:128]).to(dev)
    params=list(model.parameters())+list(head.parameters());opt=KerasAdam(params,lr=.02,eps=.1)
    groups=[[t for t in prepared['train'] if t['session']==d] for d in p['days']];sampler=DayStream(groups)
    attempt=0;history=[];best=float('inf');selected=None;latest=folder/'latest.pt'
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=False);model.load_state_dict(s['model']);head.load_state_dict(s['head']);opt.load_state_dict(s['optimizer']);sampler.restore(s['sampler'])
        attempt=s['next_attempt'];history=s['history'];best=s['best'];selected=s['selected'];torch.set_rng_state(s['torch_rng']);torch.mps.set_rng_state(s['mps_rng'])
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    start=time.monotonic();save_json(out/'status.json',dict(status='poisson-training',pid=os.getpid()))
    try:
        while attempt<10001:
            space_guard();model.train();day,batch=sampler.batch(64);opt.zero_grad(set_to_none=True)
            lr=.02*(1-opt.param_groups[0]['iterations']/10000)
            for g in opt.param_groups:g['lr']=lr
            value,parts=loss(model,head,batch,dev,day,.1,True)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite joint loss')
            value.backward();torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True);opt.step();finite_state(model,opt)
            assert all(torch.isfinite(v).all() for v in head.parameters())
            r=dict(attempt=attempt,optimizer_iteration=opt.param_groups[0]['iterations'],day=p['days'][day],ids=[t['id'] for t in batch],lr=lr,total=float(value.detach().cpu()),**parts)
            if attempt%50==0:
                score=selection(model,prepared['selection'],p['days'],dev)['per']['corpus'];r['selection_PER']=score
                if score<best:
                    best=score;selected=f'best-{attempt:05d}.pt';save_state(folder/selected,dict(model=model.state_dict(),head=head.state_dict(),attempt=attempt,selection_PER=score))
            history.append(r);attempt+=1
            if attempt%50==0 or (attempt-1)%50==0 or stop[0] or attempt==10001:
                save_state(latest,dict(model=model.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),next_attempt=attempt,history=history,best=best,selected=selected,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()))
                save_json(folder/'progress.json',dict(attempts=attempt,best_selection_PER=best,seconds=time.monotonic()-start,**parts));print(attempt,parts,'PER',best,flush=True)
            if stop[0]:save_json(out/'status.json',dict(status='stopped-resumable',attempts=attempt));return
        saved=torch.load(folder/selected,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);head.load_state_dict(saved['head'])
        metric=selection(model,prepared['selection'],p['days'],dev,folder/'selected-held-blocks');assert metric['per']['corpus']==best
        save_json(folder/'training-complete.json',dict(selected=selected,sha256=digest(folder/selected),attempts=attempt,PER=metric['per'],scope='Earlier held-block selection; independent audit and matched raw/Poisson later-day evaluation pending'))
        save_json(out/'status.json',dict(status='poisson-training-complete-audit-and-calibration-pending'))
    except Exception as e:
        save_json(out/'failure.json',dict(error=repr(e),last_checkpoint=str(latest),attempt=attempt));raise


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);main(a.parse_args().run.resolve())
