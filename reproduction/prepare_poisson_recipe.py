"""E025 freeze/pilot and resource gate; no production launch or final-day access."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
from causal_recipe_control import ROOT,INITIAL,fresh_model,objective,standardize,DayStream
from poisson_recipe import CountHead,loss
from research.audit import digest,save_json,space_guard
from research.data import load_trials


def main(out):
    assert not out.exists();space_guard();torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent CPU training fallback')
    dev=torch.device('mps');source=ROOT/'research/runs/e022-causal-recipe-s17-20261012-01'
    parent=json.loads((source/'protocol.json').read_text());audit=json.loads((source/'raw/completion-audit-01.json').read_text());assert audit['status']=='pass'
    for family in ('source_hashes','implementation_hashes','derived_hashes'):
        for p,h in parent[family].items():assert digest(p)==h
    manifest=json.loads((ROOT/'research/data/manifest.json').read_text());split=json.loads((source/'splits.json').read_text())
    train_ids={r['id'] for r in split['train']};assert not set(parent['days'])&set(parent['forbidden_final_days'])
    trials=[t for t in load_trials(manifest,parent['days'],'train') if t['id'] in train_ids];assert len(trials)==4400
    # Exact historical sampler accounting permits reuse of the completed control.
    groups=[[t for t in trials if t['session']==d] for d in parent['days']];sampler=DayStream(groups)
    history=torch.load(source/'raw/latest.pt',map_location='cpu',weights_only=False)['history']
    for r in history:
        d,b=sampler.batch(64);assert r['day']==parent['days'][d] and r['ids']==[t['id'] for t in b]
    norm=json.loads((source/'raw-normalization.json').read_text());means=np.array(norm['global_stats']['mean'])[:128]
    pilot=[dict(t,z=standardize(t['raw'],'raw',norm['per_day'][t['session']],norm['global_stats'])) for t in trials[:4]]
    out.mkdir(parents=True)
    model,opt=fresh_model(dev);model.train();reference,_=objective(model,pilot,dev,0,True);reference.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};expected_loss=float(reference.detach().cpu());del model,opt
    model,opt=fresh_model(dev);head=CountHead(means).to(dev);model.train();value,parts=loss(model,head,pilot,dev,0,0.,True)
    value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
    assert expected_loss==float(value.detach().cpu())
    for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),expected[k],rtol=0,atol=0)
    del expected,opt
    # Fresh positive-loss pilot, not warm-started control or production state.
    del model,head;model,opt=fresh_model(dev);head=CountHead(means).to(dev)
    from causal_recipe_control import KerasAdam
    opt=KerasAdam(list(model.parameters())+list(head.parameters()),lr=.02,eps=.1)
    observed=[]
    for step in range(3):
        model.train();opt.zero_grad(set_to_none=True);v,parts=loss(model,head,pilot,dev,0,.1,True)
        assert torch.isfinite(v);v.backward();params=list(model.parameters())+list(head.parameters())
        torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True);opt.step();assert all(torch.isfinite(p).all() for p in params)
        observed.append(dict(step=step,total=float(v.detach().cpu()),**parts))
    torch.save(dict(model=model.state_dict(),head=head.state_dict(),scope='Three-update technical pilot, excluded from production'),out/'pilot.pt')
    model.eval();state=torch.load(out/'pilot.pt',map_location='cpu',weights_only=False)
    head.load_state_dict(state['head']);model.load_state_dict(state['model'])
    bests=list((source/'raw').glob('best-*.pt'));reserve=sum(p.stat().st_size for p in bests)+(source/'raw/latest.pt').stat().st_size+2*2**30
    free=shutil.disk_usage(out).free;required=30*2**30+reserve
    protocol=dict(experiment='E025 / I005',status='technical-pilot-only-resource-gate',seed=17,parent_protocol_sha256=digest(source/'protocol.json'),
        initial_sha256=digest(INITIAL),raw_control=str(source/'raw/training-complete.json'),raw_control_audit_sha256=digest(source/'raw/completion-audit-01.json'),
        matched='E022 raw input/power, normalization, architecture, initial weights/RNG, sampler, KerasAdam .02->0/10000updates,epsilon.1,batch64,10001attempts,clip10,augmentations,CTC and held-block selection unchanged',
        head=dict(parameters=524800,architecture='1024->512 linear +softplus+1e-6; four128channel20mscount vectors',initialization='Zero weights; softplus-inverse bin-weighted4400training-count means for bias; CPU/MPS backbone RNG preserved'),
        reconstruction='Untransformed integer counts; final4bins of each32binpatch; ends31,35,...; no overlapping target reuse, first28warmup/last<4bins excluded; padding masked',
        objective='Original unnormalized sentence-meanCTC+originalL2+0.1*PoissonNLL; NLL mean over4subbins/128channels,sum valid frames,mean trials,exact lgamma constant',
        weight_rule='Single prospective0.1weight, no hyperparameter search or target-day tuning; negative outcome reportable',
        loss_causality='Reconstruct already-observed counts, not next-step forecasting; no assertion this identifies true rates or prevents copying noise',
        control_parity='Exact MPS zero-weight one-step loss/model parity; all10001historicalsampler batches match; no reconstruction RNG consumption',
        checkpoints='Improved full model+head retained, resumable latest full optimizer/sampler/RNG every50updates; dynamic30GiBguard; planning reserve based on historical72improvedrawcheckpoints, not worst-case guarantee',
        evaluation='Same5development days, fixed3gram,4608adapter and0/5/10/25/50/100/fullcalibration; frozen25%WERtarget; corpus/perday/worstWER/PER/CER/S-D-I/pairedconditionaluncertainty; countNLL diagnostic',
        boundary='No final-day data, no new generator/KL/predictive/power reconstruction/daygainoffset/Anscombe/square-root/v1restart. Historical public exposure andonepairedseed exploratory.',
        code_hashes={str(p):digest(p) for p in [Path(__file__),ROOT/'reproduction/poisson_recipe.py']},parent_source_hashes=parent['source_hashes'])
    save_json(out/'protocol.json',protocol);save_json(out/'pilot.json',dict(status='pass',backend='mps',zero_weight_parity='Exact original loss/model one-step',historical_sampler_batches=len(history),updates=observed,production_state='Fresh restart required; pilot not reused'))
    save_json(out/'storage-gate.json',dict(status='pass' if free>=required else 'blocked',free_GiB=free/2**30,required_GiB=required/2**30,planning_reserve_GiB=reserve/2**30,shortfall_GiB=max(0,required-free)/2**30,reason='Retain model history and30GiBfloor; no deletion authorized'))
    save_json(out/'status.json',dict(status='pilot-verified-full-training-storage-blocked' if free<required else 'pilot-verified-production-runner-pending'))
    print((out/'storage-gate.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
