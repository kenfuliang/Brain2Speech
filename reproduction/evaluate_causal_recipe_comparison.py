"""E022 selected raw/Anscombe serial calibration and fixed-decoder exports."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from causal_recipe_control import ROOT,PublishedGRU,TargetModel,moments,standardize,native_batch,objective
from run_causal_recipe_control import load
from run_preprocessing_comparison import device,save_state,state_hash
from research.audit import digest,save_json,space_guard
from research.metrics import greedy,edits,summarize
from research.experiment.evaluator import FixedDecoder

ARMS=('raw','anscombe')


def conditions(p):
    for kind in ARMS:
        for day in p['target_days']:
            for budget in p['calibration']['budgets']:
                for mode in (['normalization'] if budget==0 else ['normalization','adapter']):
                    yield kind,day,budget,mode


def prepare(run,out):
    assert not out.exists();p,_=load(run)
    parents={}
    for kind in ARMS:
        a=json.loads((run/kind/'completion-audit-01.json').read_text());assert a['status']=='pass'
        c=json.loads((run/kind/'training-complete.json').read_text());parent=run/kind/c['selected']
        assert digest(parent)==c['sha256'];parents[kind]=dict(path=str(parent),sha256=c['sha256'])
    out.mkdir(parents=True)
    save_json(out/'protocol.json',dict(parent_run=str(run),parent_protocol_sha256=digest(run/'protocol.json'),parents=parents,
        code_sha256=digest(__file__),calibration=p['calibration'],target_days=p['target_days'],WER_target=p['WER_target'],
        conditions=[list(x) for x in conditions(p)],
        metrics='Corpus/per-day/worst PER/WER/CER, exact S/D/I; paired session bootstrap; one paired seed/five previously exposed development days',
        selection='Parents fixed by earlier held-block PER; do not select/retrain on target results',
        adapter_location='After causal smoothing, before frozen historical read-in13; identical across arms',
        inference='Native MPS; eval/dropout off; trial reset; fixed3gram unchanged; no final-day access',
        resume='Per-child fast adapter/optimizer/sampler saved every20steps; each day/budget starts independently from selected parent',
        stop='30GiBfloor/nonfinite/identity mismatch; preserve partial state, no cleanup or automatic recipe changes'))
    save_json(out/'status.json',dict(status='prepared-pilot-required'))


def checked(out):
    q=json.loads((out/'protocol.json').read_text());assert digest(__file__)==q['code_sha256']
    run=Path(q['parent_run']);assert digest(run/'protocol.json')==q['parent_protocol_sha256']
    p,data=load(run)
    for parent in q['parents'].values():assert digest(parent['path'])==parent['sha256']
    return q,p,data


def parent_model(q,kind,dev):
    m=PublishedGRU().to(dev);m.load_state_dict(torch.load(q['parents'][kind]['path'],map_location='cpu',weights_only=False)['model']);m.eval();m.requires_grad_(False);return m


def pilot(out):
    q,p,data=checked(out);dev=device();parent=parent_model(q,'raw',dev);before=state_hash(parent)
    model=TargetModel(parent,p['calibration']['reference_day']);fast=list(model.fast.parameters());assert sum(x.numel() for x in fast)==4608
    norm=json.loads((Path(q['parent_run'])/'raw-normalization.json').read_text())
    trials=[dict(t,z=standardize(t['raw'],'raw',norm['global_stats'])) for t in data['train'][:2]]
    x,_,_,_=native_batch(trials[:1],dev,False)
    with torch.inference_mode():
        y=model(x).cpu(); altered=x.clone();altered[:,64:]+=10
        np.testing.assert_allclose(model(altered).cpu()[:,:8],y[:,:8],rtol=0,atol=1e-5)
        model(torch.flip(x,[1]));np.testing.assert_array_equal(model(x).cpu(),y)
    opt=torch.optim.AdamW(fast,lr=.001,weight_decay=.01);opt.zero_grad()
    loss,_=objective(model,trials,dev,0,False,False);loss=loss+.01*model.fast.regularization()
    loss.backward();torch.nn.utils.clip_grad_norm_(fast,5,error_if_nonfinite=True);opt.step()
    assert state_hash(parent)==before and all(torch.isfinite(v).all() for v in fast)
    state={k:v.detach().cpu().clone() for k,v in model.fast.state_dict().items()}
    fresh=TargetModel(parent,p['calibration']['reference_day']);fresh.fast.load_state_dict(state)
    with torch.inference_mode():np.testing.assert_array_equal(model(x).cpu(),fresh(x).cpu())
    save_json(out/'pilot.json',dict(status='pass',adapter_parameters=4608,frozen_parent='unchanged',reload='exact',trial_reset='exact',prefix_atol=1e-5,loss=float(loss.detach().cpu())))
    print('pilot passed',flush=True)


def predict(model,trials,dev,target):
    records=[];arrays={};model.eval()
    with torch.inference_mode():
        for i,t in enumerate(trials):
            x,_,n,_=native_batch([t],dev,False);y=model(x)[0,:int(n[0])].cpu().numpy()
            assert np.isfinite(y).all();phones=greedy(y)
            records.append(dict(id=t['id'],session=t['session'],text=t['text'],reference_phones=t['labels'].tolist(),predicted_phones=phones,per=edits(t['labels'].tolist(),phones)))
            arrays[f'trial_{i:04d}']=y
    save_json(target/'predictions.json',records);np.savez_compressed(target/'logits.npz',**arrays)
    save_json(target/'metrics-before-lm.json',summarize(records,repetitions=200))


def export(out):
    q,p,data=checked(out);assert json.loads((out/'pilot.json').read_text())['status']=='pass';dev=device();stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    save_json(out/'status.json',dict(status='exporting',pid=os.getpid()))
    for kind in ARMS:
        parent=parent_model(q,kind,dev);parent_hash=state_hash(parent)
        norm=json.loads((Path(q['parent_run'])/(kind+'-normalization.json')).read_text())
        for arm,day,budget,mode in conditions(p):
            if arm!=kind:continue
            target=out/kind/day/str(budget)/mode;done=target/'export-complete.json'
            if done.exists():
                seal=json.loads(done.read_text())
                for name,h in seal['artifacts_sha256'].items():assert digest(target/name)==h
                continue
            space_guard();target.mkdir(parents=True,exist_ok=True);trials=data['calibration'][day];n=len(trials) if budget=='full' else budget;subset=trials[:n]
            assert len(subset)==n and not {t['id'] for t in subset}&{t['id'] for t in data['evaluation'][day]}
            stats=moments(subset,kind) if n else norm['global_stats']
            prepared=[dict(t,z=standardize(t['raw'],kind,stats,norm['global_stats'])) for t in subset]
            evaluation=[dict(t,z=standardize(t['raw'],kind,stats,norm['global_stats'])) for t in data['evaluation'][day]]
            model=TargetModel(parent,p['calibration']['reference_day']);fast=list(model.fast.parameters())
            opt=torch.optim.AdamW(fast,lr=.001,weight_decay=.01);rng=np.random.default_rng(17);losses=[];start=time.monotonic();latest=target/'resume.pt'
            if latest.exists():
                s=torch.load(latest,map_location='cpu',weights_only=False);assert s['parent']==q['parents'][kind]['sha256']
                model.fast.load_state_dict(s['fast']);opt.load_state_dict(s['optimizer']);rng.bit_generator.state=s['rng'];losses=s['losses']
            for step in range(len(losses),100 if mode=='adapter' else 0):
                space_guard();group=[prepared[int(i)] for i in rng.integers(n,size=16)];opt.zero_grad()
                loss,_=objective(model,group,dev,0,False,False);loss=loss+.01*model.fast.regularization()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite calibration loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(fast,5,error_if_nonfinite=True);opt.step()
                assert all(torch.isfinite(v).all() for v in fast);losses.append(float(loss.detach().cpu()))
                if (step+1)%20==0 or stop[0]:save_state(latest,dict(parent=q['parents'][kind]['sha256'],fast=model.fast.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,losses=losses))
                if stop[0]:save_json(out/'status.json',dict(status='stopped-resumable',condition=[kind,day,budget,mode]));return
            assert state_hash(parent)==parent_hash
            save_state(target/'adapter.pt',model.fast.state_dict());save_json(target/'normalization.json',stats);predict(model,evaluation,dev,target)
            save_json(done,dict(parent_sha256=q['parents'][kind]['sha256'],calibration_ids=[t['id'] for t in subset],
                labels=n if mode=='adapter' else 0,observed_sentences=n,steps=len(losses),losses=losses,wall_seconds=time.monotonic()-start,
                artifacts_sha256={name:digest(target/name) for name in ['adapter.pt','normalization.json','predictions.json','logits.npz','metrics-before-lm.json']}))
            save_json(out/'progress.json',dict(last_export=[kind,day,budget,mode],complete_exports=len(list(out.glob('*/*/*/*/export-complete.json'))),total=len(q['conditions'])))
            print('exported',kind,day,budget,mode,flush=True);del model,opt;gc.collect();torch.mps.empty_cache()
        del parent;gc.collect();torch.mps.empty_cache()
    save_json(out/'status.json',dict(status='exports-complete-decoder-pending'))


def decode(out):
    q,p,_=checked(out)
    from decode_bit_full import worker_ids
    assert not worker_ids(),'Competing decoder worker'
    assert all((out/k/d/str(b)/m/'export-complete.json').exists() for k,d,b,m in conditions(p))
    save_json(out/'status.json',dict(status='decoder-loading',pid=os.getpid()))
    decoder=FixedDecoder(out/'decoder.log');owned=worker_ids();assert len(owned)==1
    save_json(out/'decoder-identity.json',dict(decoder.identity,container=owned[0]))
    try:
        for kind,day,budget,mode in conditions(p):
            target=out/kind/day/str(budget)/mode
            if (target/'decoded.json').exists():continue
            space_guard();ident=json.loads((target/'export-complete.json').read_text())
            for name,h in ident['artifacts_sha256'].items():assert digest(target/name)==h
            rows=json.loads((target/'predictions.json').read_text());response=decoder.decode(target);assert len(response['transcripts'])==len(rows)
            for r,text,ms in zip(rows,response['transcripts'],response['decode_ms']):
                r.update(decoded_text=text,wer=edits(r['text'].split(),text.split()),wer_legacy=edits(r['text'].split(' '),text.split(' ')),cer=edits(list(r['text']),list(text)),decode_ms=ms)
            save_json(target/'decoded.json',rows);save_json(target/'metrics.json',summarize(rows,repetitions=2000))
            print('decoded',kind,day,budget,mode,flush=True)
    finally:decoder.close()
    deadline=time.monotonic()+600
    while owned[0] in worker_ids() and time.monotonic()<deadline:time.sleep(5)
    assert owned[0] not in worker_ids(),'Owned decoder did not exit'
    save_json(out/'status.json',dict(status='decoded-independent-comparison-audit-pending'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path);a.add_argument('--output',type=Path,required=True);a.add_argument('--stage',choices=['prepare','pilot','export','decode'],required=True);v=a.parse_args();out=v.output.resolve()
    try:
        if v.stage=='prepare':prepare(v.run.resolve(),out)
        else:globals()[v.stage](out)
    except Exception as e:
        if out.exists():save_json(out/'failure.json',dict(stage=v.stage,error=repr(e)))
        raise
