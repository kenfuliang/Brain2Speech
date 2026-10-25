"""Architecture-faithful E008 port with original TF CPU initialization/data service.

The original timestamp seed is unavailable; stochastic Torch dropout is not a
bitwise TensorFlow trajectory. Public test selects checkpoints as in the source.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import numpy as np
import torch
from pytorch_baseline import PublishedGRU,KerasAdam

ROOT=Path(__file__).absolute().parent.parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')

class RecipeData:
    def __init__(self,output,seed):
        self.log=(output/'data-service.log').open('ab')
        self.proc=subprocess.Popen([str(ROOT/'.venv-e008-metal/bin/python'),'-u',str(ROOT/'reproduction/tf_recipe_service.py'),
                                    '--output',str(output/'tf-data-runtime'),'--seed',str(seed)],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)

    def call(self,command,**kwargs):
        self.proc.stdin.write((json.dumps(dict(command=command,**kwargs))+'\n').encode()); self.proc.stdin.flush()
        header=self.proc.stdout.read(8)
        if len(header)!=8: raise RuntimeError('TF data service exited; inspect data-service.log')
        length=struct.unpack('!Q',header)[0]
        blocks=[]
        while length:
            value=self.proc.stdout.read(min(length,1024*1024))
            if not value: raise RuntimeError('Truncated TF reply')
            blocks.append(value); length-=len(value)
        with np.load(io.BytesIO(b''.join(blocks)),allow_pickle=False) as f:
            return {k:f[k].copy() for k in f.files}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write(b'{"command":"close"}\n'); self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        self.log.close()

def data_hash(d):
    h=hashlib.sha256()
    for k in ('inputFeatures','seqClassIDs','nTimeSteps','nSeqElements','transcription'):
        h.update(d[k].tobytes())
    return h.hexdigest()

def tensors(d):
    x=torch.tensor(d['inputFeatures'],device='mps')
    targets=torch.tensor(d['seqClassIDs'].astype('int64'))-1
    lengths=torch.tensor(((d['nTimeSteps']-32)//4+1).astype('int64'))
    labels=torch.tensor(d['nSeqElements'].astype('int64'))
    if (lengths<=0).any() or (labels<=0).any(): raise ValueError('Invalid labeled trial')
    return x,targets,lengths,labels

def ctc(y,targets,lengths,labels):
    return torch.nn.functional.ctc_loss(y.cpu().log_softmax(-1).transpose(0,1),targets,lengths,labels,blank=40,reduction='none').mean().to('mps')

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]

def greedy(logits,length):
    tokens=[]; previous=40
    for token in logits[:length].argmax(-1).tolist():
        if token!=40 and token!=previous: tokens.append(token)
        previous=token
    return tokens

def validate(model,batches):
    model.eval(); rows=[]
    with torch.no_grad():
        for d in batches:
            x,target,lengths,labels=tensors(d)
            y=model(x,int(d['day'])).cpu()
            for i in range(len(x)):
                truth=target[i,:labels[i]].tolist()
                pred=greedy(y[i],int(lengths[i]))
                rows.append(dict(day=int(d['day']),reference=truth,prediction=pred,edits=distance(pred,truth),length=len(truth)))
    counts={day:sum(r['day']==day for r in rows) for day in range(19)}
    assert counts=={day:20 if day<4 else 40 for day in range(19)}, counts
    return dict(per=sum(r['edits'] for r in rows)/sum(r['length'] for r in rows),trials=len(rows),
                edits=sum(r['edits'] for r in rows),phones=sum(r['length'] for r in rows),per_day_trials=counts,rows=rows)

def save_state(path,model,opt,rng,history,next_attempt,best,seed):
    state=dict(model=model.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.get_state(),
               torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),history=history,
               next_attempt=next_attempt,best_per=best,seed=seed)
    temporary=path.with_suffix('.writing')
    torch.save(state,temporary); os.replace(temporary,path)

def restore(path,model,opt,rng,service):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
    rng.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.mps.set_rng_state(state['mps_rng'])
    for record in state['history']:
        value=service.call('train',day=record['day'])
        if data_hash(value)!=record['data_hash']: raise RuntimeError('TF stream replay differs; cannot claim exact resume')
    return state

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--last-attempt',type=int,default=10000)
    p.add_argument('--resume',action='store_true'); p.add_argument('--resume-probe',type=Path)
    p.add_argument('--require-gate',type=Path)
    p.add_argument('--resume-boundary',type=int)
    a=p.parse_args(); out=a.output.absolute()
    if not a.resume and out.exists(): raise FileExistsError(out)
    if shutil_free(out.parent)<30*1024**3: raise RuntimeError('Below 30 GiB storage guard')
    if a.last_attempt==10000 and (not a.require_gate or json.loads(a.require_gate.read_text())['status']!='passed-recipe-and-fresh-resume'):
        raise RuntimeError('Full retraining requires passed fresh-process gate')
    if a.require_gate:
        gate=json.loads(a.require_gate.read_text())
        for relative,expected in gate['source_hashes'].items():
            if sha(ROOT/relative)!=expected: raise RuntimeError(f'Gate source changed: {relative}')
    if a.resume:
        identity=json.loads((out/'run-identity.json').read_text())
        for relative,expected in identity['source_hashes'].items():
            if sha(ROOT/relative)!=expected: raise RuntimeError(f'Resume source changed: {relative}')
    out.mkdir(exist_ok=a.resume)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.backends.mps.is_available()
    torch.set_num_threads(8); torch.manual_seed(a.seed)
    service=RecipeData(out,a.seed)
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        initial=service.call('initial')
        cfg=json.loads(str(initial['configuration']))
        assert int(initial['gru_implementation'])==2
        assert cfg['batchSize']==64 and cfg['batchesPerVal']==50 and cfg['learnRateDecaySteps']==10000
        model=PublishedGRU(); model.load_tf(initial); model.to('mps')
        opt=KerasAdam(model.parameters()); rng=np.random.RandomState(a.seed)
        history=[]; best=1.; start_attempt=0
        if a.resume:
            state=restore(out/'latest.pt',model,opt,rng,service)
            assert state['seed']==a.seed
            history=state['history']; best=state['best_per']; start_attempt=state['next_attempt']
        elif not a.resume_probe:
            identity=dict(experiment='E008',seed=a.seed,configuration=cfg,torch=torch.__version__,tensorflow=str(initial['tensorflow']),
                parameter_count=sum(v.numel() for v in model.parameters()),initialization='Original random-initialized TF CPU constructor; no released trained checkpoint',
                source_hashes={str(x.relative_to(ROOT)):sha(x) for x in [Path(__file__),ROOT/'reproduction/pytorch_baseline.py',ROOT/'reproduction/tf_recipe_service.py',ROOT/'NeuralDecoder/neuralDecoder/models.py',ROOT/'NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py',ROOT/'NeuralDecoder/neuralDecoder/datasets/speechDataset.py',ROOT/'workspace/derived/rnns/baselineRelease/args.yaml']},
                tfrecord_hashes={str(x.relative_to(ROOT)):sha(x) for s in cfg['dataset']['sessions'] for split in ('train','test') for x in sorted((ROOT/'workspace/derived/tfRecords'/s/split).glob('*.tfrecord'))},
                deviations=['PyTorch/MPS GRU and dropout rather than TF GPU; distribution-faithful, not bitwise stochastic replay','Explicit seed rather than unavailable original timestamp','Explicit CPU CTC with differentiable transfer','Retain all best checkpoints; atomic rolling latest runtime state instead of upstream pruning'],
                selection='Greedy PER on all 680 public-test trials (first four days 20 each, remaining 15 days 40 each) every 50 attempts including attempt 0; later WFST benchmark uses last 600; selection/evaluation overlap disclosed',
                last_attempt=a.last_attempt,full_training=a.last_attempt==10000)
            write(out/'run-identity.json',identity)
            save_state(out/'initial.pt',model,opt,rng,history,0,best,a.seed)
        if a.resume_probe:
            state=restore(a.resume_probe,model,opt,rng,service)
            history=state['history']; best=state['best_per']; start_attempt=state['next_attempt']
        # One fixed training-only probe validates exact model restoration separately from CTC.
        probe=np.load(ROOT/'reproduction/results/retraining/pytorch-fixture-20261007-01.npz')['input']
        model.eval()
        with torch.no_grad(): probe_logits=model(torch.tensor(probe,device='mps')).cpu().numpy()
        if a.resume_probe:
            np.save(out/'restored-logits.npy',probe_logits)
        # Cache original validation batches; zero augmentation, unchanged preprocessing.
        first=service.call('validation',index=0)
        batches=[first]+[service.call('validation',index=i) for i in range(1,int(first['batches']))]
        journal_path=out/(f'training-resume-{time.time_ns()}.jsonl' if a.resume else 'training.jsonl')
        journal=journal_path.open('x')
        for attempt in range(start_attempt,a.last_attempt+1):
            if stop[0] or shutil_free(out)<30*1024**3:
                save_state(out/'latest.pt',model,opt,rng,history,attempt,best,a.seed)
                write(out/'status.json',dict(status='paused-storage' if not stop[0] else 'stopped-signal',next_attempt=attempt,pid=os.getpid()))
                return
            t=time.monotonic(); day=int(np.argwhere(rng.multinomial(1,cfg['dataset']['datasetProbability']))[0][0])
            d=service.call('train',day=day); record=dict(day=day,data_hash=data_hash(d)); history.append(record)
            model.train(); opt.zero_grad(set_to_none=True)
            x,targets,lengths,labels=tensors(d)
            y=model(x,day); pred_loss=ctc(y,targets,lengths,labels); loss=pred_loss+model.regularization(day)
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            finite=all(v.grad is None or bool(torch.isfinite(v.grad).all().item()) for v in model.parameters())
            iterations=opt.param_groups[0]['iterations']; lr=.02*(1-min(iterations,10000)/10000)
            opt.param_groups[0]['lr']=lr
            if finite: opt.step()
            row=dict(attempt=attempt,session=cfg['dataset']['sessions'][day],**record,loss=float(loss.detach().cpu()),prediction_loss=float(pred_loss.detach().cpu()),gradient_norm=float(norm.detach().cpu()),lr=lr,update_applied=finite,optimizer_iterations=opt.param_groups[0]['iterations'],seconds=time.monotonic()-t)
            journal.write(json.dumps(row)+'\n'); journal.flush(); print(json.dumps(row),flush=True)
            write(out/'status.json',dict(status='training',pid=os.getpid(),last_batch=row,best_validation_per=best))
            if a.resume_boundary is not None and attempt+1==a.resume_boundary:
                save_state(out/f'resume-boundary-{attempt+1}.pt',model,opt,rng,history,attempt+1,best,a.seed)
            if attempt%50==0:
                result=validate(model,batches); write(out/f'validation-{attempt:05d}.json',result)
                if result['per']<best:
                    best=result['per']; save_state(out/f'best-{attempt:05d}.pt',model,opt,rng,history,attempt+1,best,a.seed)
                    write(out/'selected.json',dict(attempt=attempt,per=best,checkpoint=f'best-{attempt:05d}.pt'))
                save_state(out/'latest.pt',model,opt,rng,history,attempt+1,best,a.seed)
                print(json.dumps(dict(validation_attempt=attempt,per=result['per'],trials=result['trials'],best=best)),flush=True)
            if attempt==a.last_attempt:
                save_state(out/'latest.pt',model,opt,rng,history,attempt+1,best,a.seed)
                model.eval()
                with torch.no_grad(): np.save(out/'final-probe-logits.npy',model(torch.tensor(probe,device='mps')).cpu().numpy())
        journal.close()
        completed=dict(status='training-complete-evaluation-required',attempts=a.last_attempt+1,best_per=best,wer=None,seed=a.seed)
        write(out/'completion.json',completed); write(out/'status.json',completed)
    finally:
        service.close()

def shutil_free(path):
    import shutil
    return shutil.disk_usage(path).free

if __name__=='__main__': main()
