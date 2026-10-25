"""E021 serial, resumable count-transform comparison; v1 remains untouched."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch

from preprocessing_comparison import ROOT, TRANSFORMS, ComparisonGRU, fit_stats, preprocess, split_rows, ctc
from research.audit import digest,save_json,space_guard
from research.data import load_trials
from research.metrics import edits,greedy,summarize
from research.experiment.evaluator import FixedDecoder,SCORING,IMAGE_ID

BUDGETS=(0,5,10,25,50,100,'full')


def save_state(path,payload):
    space_guard();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(path)


def seed():
    torch.manual_seed(17);np.random.seed(17)
    if torch.backends.mps.is_available():torch.mps.manual_seed(17)


def device():
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; do not silently start hours of CPU training')
    torch.set_num_threads(4);return torch.device('mps')


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def load_data(manifest,split):
    by_id={}
    for partition,days in [('train',manifest['folds']['initial_train']+manifest['folds']['development']),
                           ('test',manifest['folds']['development'])]:
        for t in load_trials(manifest,days,partition):by_id[t['id']]=t
    return {key:({day:[by_id[r['id']] for r in rows] for day,rows in value.items()}
                if isinstance(value,dict) else [by_id[r['id']] for r in value])for key,value in split.items()}


def prepare(out):
    space_guard()
    manifest=json.loads((ROOT/'research/data/manifest.json').read_text());split=split_rows(manifest)
    data=load_data(manifest,split);invalid=[]
    # Eligibility is identical across all transforms and budgets, never silently filtered.
    def eligible(t):
        n=(len(t['raw'])-32)//4+1;y=t['labels'];minimum=len(y)+int(np.sum(y[1:]==y[:-1]))
        if n<minimum:invalid.append(dict(id=t['id'],frames=n,minimum=minimum));return False
        return True
    for key,value in data.items():
        if isinstance(value,dict):
            for day,rows in value.items():value[day]=[t for t in rows if eligible(t)]
        else:data[key]=[t for t in value if eligible(t)]
    if invalid:raise RuntimeError('CTC eligibility requires explicit protocol decision: '+str(invalid[:3]))
    sources=[ROOT/'research/data/manifest.json']
    for partition,days in [('train',manifest['folds']['initial_train']+manifest['folds']['development']),('test',manifest['folds']['development'])]:
        for day in days:
            p=ROOT/manifest['sessions'][day][partition]['path']
            if digest(p)!=manifest['sessions'][day][partition]['sha256']:raise ValueError('Source identity mismatch')
            sources.append(p)
    code=[ROOT/'reproduction'/n for n in ['preprocessing_comparison.py','run_preprocessing_comparison.py','pytorch_baseline.py']]
    code += [ROOT/'research'/n for n in ['data.py','metrics.py','audit.py','experiment/evaluator.py','experiment/wfst_worker.py']]
    protocol=dict(experiment='E021',date_local='2026-10-12',seed=17,transforms=list(TRANSFORMS),
        architecture='Fresh shared Willett-style five forward GRUs,1024 units,32-bin input windows,stride4; shared256 projection; gain/bias+rank8 adapter',
        causal='Accepted4ms acquisition delay; trailing9-bin sigma2 smoothing; recurrent trial reset; no downstream lookahead',
        normalization='Equal-trial moments fit initial training subset only, independently after each count transform; fixed for all target-day budgets',
        power='log1p, same for all arms',weights='Fresh paired initialization, no E017 pretrained weights',
        selection='Last whole training block of each initial day; target days never select neural checkpoints',
        calibration='Independent day and budget children of the same selected parent; shared chronological prefixes; only4608 adapter parameters update',
        budgets=list(BUDGETS),training=dict(steps=10000,batch=16,lr_start=.001,lr_end=.0003,
        optimizer='AdamW',weight_decay=.01,gradient_clip=5.,validation_every=100,sampling='Uniform day then uniform trials with replacement'),
        adaptation=dict(steps=100,lr=.001,batch=16,weight_decay=.01,regularization=.01,
        frozen_dropout='Disabled; backbone eval with adapter gradients',gain_log_bounds=[-.7,.7],offset_bounds=[-2.,2.]),
        WER_target=.25,target_status='Prespecified exploratory target; report not reached rather than extrapolate',
        decoding=dict(image_id=IMAGE_ID,scoring=SCORING,word_tokens='standard whitespace; legacy literal-space also retained'),
        inference='Single participant/seed, five later days; historically inspected public data; no untouched confirmation claim',
        forbidden_final_days=manifest['folds']['final'],v1='Preserved,paused',minimum_free_GiB=30,
        source_hashes={str(p):digest(p)for p in sources},implementation_hashes={str(p):digest(p)for p in code},
        accounting={k:({day:len(rows)for day,rows in v.items()}if isinstance(v,dict)else len(v))for k,v in split.items()})
    out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',protocol);save_json(out/'splits.json',split)
    for kind in TRANSFORMS:save_json(out/(kind+'-normalization.json'),fit_stats(data['train'],kind))
    save_json(out/'status.json',dict(status='prepared',WER='Not measured',PER='Not measured'))
    print('Prepared E021',protocol['accounting'],flush=True)


def checked(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for key in ['source_hashes','implementation_hashes']:
        for p,h in protocol[key].items():
            if digest(p)!=h:raise ValueError('Frozen input changed: '+p)
    m=json.loads((ROOT/'research/data/manifest.json').read_text())
    return protocol,load_data(m,json.loads((out/'splits.json').read_text()))


def prepared(data,stats):
    def apply(rows):return [dict(t,x=preprocess(t['raw'],stats))for t in rows]
    return {k:({d:apply(rows)for d,rows in v.items()}if isinstance(v,dict)else apply(v))for k,v in data.items()}


def predictions(model,trials,dev,folder=None):
    model.eval();rows=[];logits={}
    with torch.inference_mode():
        for i,t in enumerate(trials):
            y=model(torch.from_numpy(t['x'][None]).to(dev))[0].cpu().numpy()
            if not np.isfinite(y).all():raise FloatingPointError('Nonfinite inference')
            p=greedy(y);rows.append(dict(id=t['id'],session=t['session'],text=t['text'],
                reference_phones=t['labels'].tolist(),predicted_phones=p,per=edits(t['labels'].tolist(),p)))
            logits[f'trial_{i:04d}']=y
    metrics=summarize(rows,repetitions=200)
    if folder:
        folder.mkdir(parents=True,exist_ok=True);save_json(folder/'predictions.json',rows)
        np.savez_compressed(folder/'logits.npz',**logits);save_json(folder/'metrics-before-lm.json',metrics)
    return metrics


def pilot(out):
    protocol,data=checked(out);dev=device();results={};initial=[]
    for kind in TRANSFORMS:
        seed();stats=json.loads((out/(kind+'-normalization.json')).read_text())
        trials=[dict(t,x=preprocess(t['raw'],stats))for t in data['train'][:4]]
        model=ComparisonGRU().to(dev);initial.append(state_hash(model));opt=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001)
        losses=[]
        for step in range(4):
            model.train();opt.zero_grad();loss=ctc(model,trials,dev)
            if not torch.isfinite(loss):raise FloatingPointError('Pilot loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        model.eval();x=torch.from_numpy(trials[0]['x'][None]).to(dev)
        with torch.inference_mode():
            y=model(x).cpu();future=x.clone();future[:,64:]+=10
            assert torch.equal(y[:,:8],model(future).cpu()[:,:8])
            assert torch.equal(y[:,:8],model(x[:,:64]).cpu()[:,:8])
            model(torch.flip(x,[1]));assert torch.equal(y,model(x).cpu())
        save_state(out/(kind+'-pilot.pt'),dict(model=model.state_dict()))
        reloaded=ComparisonGRU().to(dev);reloaded.load_state_dict(torch.load(out/(kind+'-pilot.pt'),map_location='cpu',weights_only=False)['model']);reloaded.eval()
        with torch.inference_mode():assert torch.equal(y,reloaded(x).cpu())
        reloaded.calibrate_only();before={k:v.cpu().clone()for k,v in reloaded.state_dict().items()}
        fast=[p for p in reloaded.parameters()if p.requires_grad];assert sum(p.numel()for p in fast)==4608
        optimizer=torch.optim.AdamW(fast,lr=.001);optimizer.zero_grad();loss=ctc(reloaded,trials[:1],dev)
        loss.backward();torch.nn.utils.clip_grad_norm_(fast,5.,error_if_nonfinite=True);optimizer.step()
        assert all(torch.equal(before[k],v.cpu())for k,v in reloaded.state_dict().items()if k not in ['adapters.0.log_gain','adapters.0.offset','adapters.0.u','adapters.0.v'])
        results[kind]=dict(losses=losses,adapter_parameters=4608,prefix_and_reload='exact',frozen_backbone='unchanged')
        del model,reloaded,opt,optimizer;gc.collect();torch.mps.empty_cache()
    assert len(set(initial))==1
    save_json(out/'pilot.json',dict(status='pass',arms=results,matched_initialization_sha256=initial[0],backend='mps'))
    print('Pilot passed',results,flush=True)


def train(out):
    protocol,data=checked(out);assert json.loads((out/'pilot.json').read_text())['status']=='pass'
    dev=device();stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    for kind in TRANSFORMS:
        folder=out/kind;folder.mkdir(exist_ok=True)
        if (folder/'training-complete.json').exists():continue
        seed();stats=json.loads((out/(kind+'-normalization.json')).read_text());p=prepared({'train':data['train'],'selection':data['selection']},stats)
        model=ComparisonGRU().to(dev);opt=torch.optim.AdamW([v for v in model.parameters()if v.requires_grad],lr=.001,weight_decay=.01)
        rng=np.random.default_rng(17);groups={d:[t for t in p['train']if t['session']==d]for d in sorted({t['session']for t in p['train']})}
        days=list(groups);step=0;best=float('inf');history=[];latest=folder/'latest.pt';selected=None
        if latest.exists():
            payload=torch.load(latest,map_location='cpu',weights_only=False);model.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer'])
            step=payload['step'];best=payload['best'];history=payload['history'];selected=payload['selected']
            rng.bit_generator.state=payload['rng'];torch.set_rng_state(payload['torch_rng']);torch.mps.set_rng_state(payload['mps_rng'])
        began=time.monotonic()
        while step<protocol['training']['steps']:
            space_guard();model.train();day=days[int(rng.integers(len(days)))];g=groups[day]
            batch=[g[int(i)]for i in rng.integers(len(g),size=16)]
            lr=.001+(.0003-.001)*step/(protocol['training']['steps']-1)
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad();loss=ctc(model,batch,dev)
            if not torch.isfinite(loss):raise FloatingPointError('Training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();step+=1
            history.append(dict(step=step,day=day,ids=[t['id']for t in batch],ctc=float(loss.detach()),lr=lr))
            if step%100==0:
                metric=predictions(model,p['selection'],dev);score=metric['per']['macro_day']
                if score<best:
                    best=score;selected=f'best-{step:05d}.pt';save_state(folder/selected,dict(model=model.state_dict(),step=step,selection_per=best))
                history[-1]['selection_per']=score
            if step%20==0 or stop[0] or step==protocol['training']['steps']:
                save_state(latest,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,selected=selected,
                    history=history,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()))
                save_json(folder/'progress.json',dict(step=step,loss=history[-1]['ctc'],selection_per=None if best==float('inf')else best,elapsed_seconds_this_process=time.monotonic()-began))
                print(kind,step,history[-1]['ctc'],flush=True)
            if stop[0]:save_json(out/'status.json',dict(status='stopped-resumable',arm=kind,step=step));return
        save_json(folder/'training-complete.json',dict(selected=selected,sha256=digest(folder/selected),steps=step,selection_per=best))
        del model,opt,p;gc.collect();torch.mps.empty_cache()
    save_json(out/'status.json',dict(status='training-complete-calibration-and-WER-pending'))


def calibrate(out):
    protocol,data=checked(out);dev=device()
    for kind in TRANSFORMS:
        folder=out/kind;selected=json.loads((folder/'training-complete.json').read_text());parent=folder/selected['selected']
        assert digest(parent)==selected['sha256'];stats=json.loads((out/(kind+'-normalization.json')).read_text())
        p=prepared({'calibration':data['calibration'],'evaluation':data['evaluation']},stats)
        for day,trials in p['calibration'].items():
            for budget in BUDGETS:
                n=len(trials)if budget=='full'else budget;target=folder/day/str(budget)
                if (target/'export-complete.json').exists():continue
                space_guard();seed();model=ComparisonGRU().to(dev);model.load_state_dict(torch.load(parent,map_location='cpu',weights_only=False)['model'])
                model.calibrate_only();fast=[v for v in model.parameters()if v.requires_grad];optimizer=torch.optim.AdamW(fast,lr=.001,weight_decay=.01)
                subset=trials[:n];rng=np.random.default_rng(17);losses=[];start=time.monotonic()
                for step in range(100 if n else 0):
                    group=[subset[int(i)]for i in rng.integers(n,size=16)]
                    optimizer.zero_grad();loss=ctc(model,group,dev)+.01*model.adapters[0].regularization().cpu()
                    if not torch.isfinite(loss):raise FloatingPointError('Adaptation loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(fast,5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
                target.mkdir(parents=True,exist_ok=True)
                save_state(target/'adapter.pt',{k:v.cpu()for k,v in model.adapters[0].state_dict().items()if not k.startswith('shared.')})
                predictions(model,p['evaluation'][day],dev,target)
                save_json(target/'export-complete.json',dict(parent_sha256=selected['sha256'],calibration_ids=[t['id']for t in subset],
                    budget=budget,labels=n,steps=len(losses),losses=losses,adaptation_seconds=time.monotonic()-start,
                    artifacts_sha256={name:digest(target/name)for name in ['adapter.pt','predictions.json','logits.npz','metrics-before-lm.json']}))
                print('exported',kind,day,budget,flush=True);del model,optimizer;gc.collect();torch.mps.empty_cache()
    save_json(out/'status.json',dict(status='exports-complete-WER-pending'))


def decode(out):
    protocol,_=checked(out)
    from decode_bit_full import worker_ids
    assert not worker_ids(),'Do not start competing WFST workers'
    decoder=FixedDecoder(out/'decoder.log');owned=worker_ids();assert len(owned)==1
    save_json(out/'decoder-identity.json',dict(decoder.identity,container=owned[0]))
    try:
        for kind in TRANSFORMS:
            for day in protocol['accounting']['evaluation']:
                for budget in BUDGETS:
                    target=out/kind/day/str(budget)
                    if (target/'decoded.json').exists():continue
                    ident=json.loads((target/'export-complete.json').read_text())
                    for name,h in ident['artifacts_sha256'].items():assert digest(target/name)==h
                    rows=json.loads((target/'predictions.json').read_text());response=decoder.decode(target)
                    assert len(response['transcripts'])==len(rows)
                    for row,text,ms in zip(rows,response['transcripts'],response['decode_ms']):
                        row.update(decoded_text=text,wer=edits(row['text'].split(),text.split()),
                            wer_legacy=edits(row['text'].split(' '),text.split(' ')),cer=edits(list(row['text']),list(text)),decode_ms=ms)
                    save_json(target/'decoded.json',rows);save_json(target/'metrics.json',summarize(rows,repetitions=2000))
    finally:decoder.close()
    for attempt in range(120):
        if owned[0]not in worker_ids():break
        time.sleep(5)
    assert owned[0]not in worker_ids(),'Owned worker exit required'
    save_json(out/'status.json',dict(status='decoded-independent-comparison-audit-pending',WER_target=.25))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','pilot','train','calibrate','decode'],required=True)
    a=p.parse_args();globals()[a.stage](a.output.resolve())
