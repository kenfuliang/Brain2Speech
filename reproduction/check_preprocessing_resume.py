"""Fresh-process GPU replay of the E021 training checkpoint mechanism."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from run_preprocessing_comparison import checked,device,save_state
from preprocessing_comparison import ComparisonGRU,preprocess,ctc
from research.audit import save_json,digest


def hashed(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):h.update(str((x.shape,x.dtype)).encode());h.update(x.cpu().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def update(model,opt,trials,dev):
    model.train();opt.zero_grad();loss=ctc(model,trials,dev)
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
    return dict(loss=float(loss.detach()),model=hashed(model.state_dict()),optimizer=hashed(opt.state_dict()),
                torch_rng=hashed(torch.get_rng_state()),mps_rng=hashed(torch.mps.get_rng_state()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--child',action='store_true');args=parser.parse_args();run=args.run.resolve();dev=device()
    if args.child:
        payload=torch.load(run/'resume-gate-state.pt',map_location='cpu',weights_only=False)
        model=ComparisonGRU().to(dev);model.load_state_dict(payload['model']);opt=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001,weight_decay=.01);opt.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng']);torch.mps.set_rng_state(payload['mps_rng'])
        save_json(run/'resume-gate-observed.json',update(model,opt,payload['trials'],dev));return
    _,data=checked(run);stats=json.loads((run/'raw-normalization.json').read_text())
    trials=[dict(t,x=preprocess(t['raw'],stats))for t in data['train'][:4]]
    model=ComparisonGRU().to(dev);model.load_state_dict(torch.load(run/'raw-pilot.pt',map_location='cpu',weights_only=False)['model'])
    opt=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001,weight_decay=.01)
    update(model,opt,trials,dev)
    save_state(run/'resume-gate-state.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),trials=trials,
        torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()))
    expected=update(model,opt,trials,dev);save_json(run/'resume-gate-expected.json',expected)
    del model,opt,data;torch.mps.empty_cache()
    subprocess.run([sys.executable,__file__,'--run',str(run),'--child'],check=True)
    observed=json.loads((run/'resume-gate-observed.json').read_text());assert expected==observed
    save_json(run/'resume-gate.json',dict(status='pass',fresh_process=True,exact=['loss','model','AdamW','CPU RNG','MPS RNG'],
        implementation_sha256=digest(__file__),state_sha256=digest(run/'resume-gate-state.pt')))
    print('Fresh-process resume passed',flush=True)


if __name__=='__main__':main()
