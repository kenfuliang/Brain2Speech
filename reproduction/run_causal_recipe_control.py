"""E022 raw-first training gate; never opens the five designated final days."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch

from causal_recipe_control import ROOT,INITIAL,KINDS,PublishedGRU,KerasAdam,DayStream,split_rows,fit_normalizers,standardize,smooth,native_batch,objective,fresh_model,finite_state
from run_preprocessing_comparison import load_data,save_state,device
from research.audit import digest,save_json,space_guard
from research.metrics import edits,greedy,summarize
from research.experiment.evaluator import SCORING,IMAGE_ID


def load(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for family in ('source_hashes','implementation_hashes','derived_hashes'):
        for path,h in protocol[family].items():
            if digest(path)!=h:raise ValueError('Frozen input changed: '+path)
    manifest=json.loads((ROOT/'research/data/manifest.json').read_text())
    assert not set(protocol['days'])&set(manifest['folds']['final'])
    return protocol,load_data(manifest,json.loads((out/'splits.json').read_text()))


def prepare(out):
    assert not out.exists();space_guard()
    manifest=json.loads((ROOT/'research/data/manifest.json').read_text());split=split_rows(manifest);data=load_data(manifest,split)
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    assert initial['next_attempt']==0 and initial['seed']==17 and not initial['history'] and not initial['optimizer']['state']
    days=manifest['folds']['initial_train']
    assert len(days)==14 and len(manifest['folds']['development'])==5
    rows=split['train']+split['selection']+sum(split['calibration'].values(),[])+sum(split['evaluation'].values(),[])
    assert not set(manifest['folds']['final'])&{r['session']for r in rows}
    # All arm eligibility is identical; missing CTC paths are a stop, not a deletion.
    for group in [data['train'],data['selection'],*data['calibration'].values(),*data['evaluation'].values()]:
        for t in group:
            y=t['labels'];assert (t['bins']-32)//4+1>=len(y)+int(np.sum(y[1:]==y[:-1]))
    sources=[ROOT/'research/data/manifest.json',INITIAL]
    for part,selected in [('train',days+manifest['folds']['development']),('test',manifest['folds']['development'])]:
        for day in selected:
            path=ROOT/manifest['sessions'][day][part]['path'];assert digest(path)==manifest['sessions'][day][part]['sha256'];sources.append(path)
    out.mkdir(parents=True);save_json(out/'splits.json',split)
    for kind in KINDS:save_json(out/(kind+'-normalization.json'),fit_normalizers(data['train'],kind))
    code=[ROOT/'reproduction'/n for n in ['causal_recipe_control.py','run_causal_recipe_control.py','pytorch_baseline.py','preprocessing_comparison.py','run_preprocessing_comparison.py']]
    code += [ROOT/'research'/n for n in ['audit.py','data.py','metrics.py','experiment/evaluator.py','experiment/wfst_worker.py']]
    protocol=dict(experiment='E022',date_local='2026-10-12',seed=17,days=days,target_days=manifest['folds']['development'],
        forbidden_final_days=manifest['folds']['final'],kinds=list(KINDS),
        initialization='Actual zero-update untrained E017 ancestor, not a trained checkpoint; same weights and RNG across arms',
        architecture='PublishedGRU unchanged: five1024 causal GRUs,32-bin patches/stride4,41head,19 input slots; only14 historical slots receive data',
        training=dict(attempts=10001,batch=64,optimizer='KerasAdam',eps_hat=.1,lr=.02,linear_decay_updates=10000,
            clip=10,CTC='Unnormalized sentence mean',L2='1e-5 input kernels and selected-day projection',
            noise_SD=1.,channel_offset_SD=.2,noise_order='After normalization, before causal smoothing',
            validation='After attempts0,50,...10000; minimum corpus PER on480 earlier held-block trials',
            sampler='Uniform14days; whole-session shuffle then repeat before batch64; NumPy RNG, not exact TF stream'),
        normalizers='Bin-weighted per historical day, fitted only4400 initial training subset; zero-SD fallback to training-global SD; no power log',
        inference='Trailing9-bin Gaussian sigma2;4ms accepted acquisition; recurrent reset; frame timestamps at patch ends',
        raw_gate=dict(metric='Initial held-block corpus PER',threshold=.30,when='Full10001-attempt budget selected checkpoint',
            purpose='Useful phonetic raw control, not equivalence to offline17.43%PER or future-day evidence'),
        comparison='Only count transform varies; power, seed, initial weights, normalization algorithm, sampler, augmentation, architecture, optimizer, budget, selection and decoder fixed',
        calibration=dict(budgets=[0,5,10,25,50,100,'full'],reference_day=13,
            normalization='n>0 moments use only same n earlier target train observations; n0 uses training-global; report unlabeled observation time',
            mode='4608 gain/bias+rank8 parameters only; parent and latest historical input map frozen; each child starts from selected parent',
            optimizer='AdamW',lr=.001,steps=100,batch=16,clip=5,regularization=.01,weight_decay=.01,
            normalization_only='Same preceding prefix and zero label updates, separately evaluated'),
        WER_target=.25,decoding=dict(image_id=IMAGE_ID,scoring=SCORING,tokenization='standard whitespace; legacy retained'),
        boundaries='Historical public data exposure, one participant/seed, five later days; no untouched independent confirmation claim',
        departures='Causal smoothing; past-only daily/global/prefix moments;14past-day cohort and480held-block selection; native shuffle/RNG; new-day small adapter',
        accounting={k:({d:len(g)for d,g in v.items()}if isinstance(v,dict)else len(v))for k,v in split.items()},
        source_hashes={str(p):digest(p)for p in sources},implementation_hashes={str(p):digest(p)for p in code},
        derived_hashes={str(out/n):digest(out/n)for n in ['splits.json','raw-normalization.json','sqrt-normalization.json','anscombe-normalization.json']},
        storage='30GiB guard; improved model-only checkpoints preserved; mutable latest model/optimizer/RNG; no deletion',
        v1='Paused unchanged; E021sealed unchanged')
    save_json(out/'protocol.json',protocol);save_json(out/'status.json',dict(status='prepared-raw-gate-required',WER='Not measured',PER='Not measured'))
    print('Prepared causal raw recovery',protocol['accounting'],flush=True)


def processed(data,norm):
    output={}
    for key in ('train','selection'):
        output[key]=[dict(t,z=standardize(t['raw'],norm['kind'],norm['per_day'][t['session']],norm['global_stats']))for t in data[key]]
    return output


def selection(model,trials,days,dev,output=None):
    model.eval();rows=[];saved={}
    with torch.inference_mode():
        for day_index,day in enumerate(days):
            group=[t for t in trials if t['session']==day]
            for start in range(0,len(group),64):
                batch=group[start:start+64];x,_,n,_=native_batch(batch,dev,False);y=model(x,day_index).cpu().numpy()
                if not np.isfinite(y).all():raise FloatingPointError('Nonfinite selection logits')
                for i,t in enumerate(batch):
                    logit=y[i,:int(n[i])];phones=greedy(logit);rows.append(dict(id=t['id'],session=day,text=t['text'],
                        reference_phones=t['labels'].tolist(),predicted_phones=phones,per=edits(t['labels'].tolist(),phones)))
                    saved[f'trial_{len(rows)-1:04d}']=logit
    metric=summarize(rows,repetitions=200)
    if output:
        output.mkdir(exist_ok=False);save_json(output/'predictions.json',rows);np.savez_compressed(output/'logits.npz',**saved);save_json(output/'metrics.json',metric)
    return metric


def train(out,kind):
    protocol,data=load(out);assert json.loads((out/'pilot.json').read_text())['status']=='pass'
    assert json.loads((out/'resume-gate.json').read_text())['status']=='pass'
    if kind!='raw':assert json.loads((out/'raw/raw-performance-gate.json').read_text())['status']=='pass'
    dev=device();folder=out/kind;folder.mkdir(exist_ok=True)
    if (folder/'training-complete.json').exists():return
    norm=json.loads((out/(kind+'-normalization.json')).read_text());p=processed(data,norm);days=protocol['days']
    groups=[[t for t in p['train']if t['session']==d]for d in days];sampler=DayStream(groups)
    model,opt=fresh_model(dev);attempt=0;best=float('inf');selected=None;history=[];latest=folder/'latest.pt'
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=False);model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.restore(s['sampler'])
        attempt=s['next_attempt'];best=s['best'];selected=s['selected'];history=s['history'];torch.set_rng_state(s['torch_rng']);torch.mps.set_rng_state(s['mps_rng'])
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    start=time.monotonic();save_json(out/'status.json',dict(status='training',arm=kind,pid=os.getpid()))
    try:
        while attempt<10001:
            space_guard();model.train();day,batch=sampler.batch(64);opt.zero_grad(set_to_none=True)
            lr=.02*(1-opt.param_groups[0]['iterations']/10000)
            for g in opt.param_groups:g['lr']=lr
            loss,ctc=objective(model,batch,dev,day,True)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step();finite_state(model,opt)
            record=dict(attempt=attempt,optimizer_iteration=opt.param_groups[0]['iterations'],day=days[day],ids=[t['id']for t in batch],CTC=ctc,loss=float(loss.detach().cpu()),lr=lr)
            if attempt%50==0:
                metric=selection(model,p['selection'],days,dev);score=metric['per']['corpus'];record['selection_PER']=score
                if score<best:
                    best=score;selected=f'best-{attempt:05d}.pt';save_state(folder/selected,dict(model=model.state_dict(),attempt=attempt,selection_PER=score))
                print(kind,attempt,'CTC',ctc,'selection PER',score,flush=True)
            history.append(record);attempt+=1
            if attempt%50==0 or stop[0] or attempt==10001 or (attempt-1)%50==0:
                save_state(latest,dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),next_attempt=attempt,
                    best=best,selected=selected,history=history,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()))
                save_json(folder/'progress.json',dict(attempts=attempt,CTC=ctc,best_selection_PER=best,elapsed_seconds_this_process=time.monotonic()-start))
            if stop[0]:save_json(out/'status.json',dict(status='stopped-resumable',arm=kind,attempts=attempt));return
    except Exception as error:
        save_json(out/'failure.json',dict(status='failed-stop',arm=kind,error=str(error),last_sealed_checkpoint=str(latest)));raise
    model.load_state_dict(torch.load(folder/selected,map_location='cpu',weights_only=False)['model'])
    selected_metric=selection(model,p['selection'],days,dev,folder/'selected-held-blocks')
    assert selected_metric['per']['corpus']==best
    completion=dict(selected=selected,sha256=digest(folder/selected),attempts=attempt,PER=selected_metric['per'])
    save_json(folder/'training-complete.json',completion)
    if kind=='raw':save_json(folder/'raw-performance-gate.json',dict(status='pass'if best<=.30 else'fail',PER=best,threshold=.30,
        budget=10001,checkpoint_sha256=completion['sha256'],scope='Earlier held blocks; not later-day WER or paper equivalence'))
    save_json(out/'status.json',dict(status='arm-training-complete',arm=kind,raw_gate='pass'if best<=.30 else'fail'if kind=='raw'else'not applicable'))
    print('completed',kind,best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','train'],required=True);p.add_argument('--kind',choices=KINDS,default='raw');a=p.parse_args()
    if a.stage=='prepare':prepare(a.run.resolve())
    else:train(a.run.resolve(),a.kind)
