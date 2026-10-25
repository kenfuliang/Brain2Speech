"""CPU/MPS recurrent gradients with identical weights; diagnostic only."""
import argparse
import copy
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    import numpy as np
    import torch
    assert torch.backends.mps.is_available()
    rows=[]
    for units in (4,32,1024):
        torch.manual_seed(17)
        source=torch.nn.GRU(3,units,batch_first=True)
        x=torch.tensor(np.random.RandomState(17).randn(2,8,3).astype('float32')*.2)
        results=[]
        for device in ('cpu','mps'):
            model=copy.deepcopy(source).to(device)
            t=time.monotonic()
            y,_=model(x.to(device))
            loss=(y*y).sum()
            loss.backward()
            results.append({'output':y.detach().cpu().numpy().copy(),
                            'loss':float(loss.detach().cpu()),
                            'grads':[v.grad.detach().cpu().numpy().copy() for v in model.parameters()],
                            'seconds':time.monotonic()-t})
        differences=[{'parameter':i,'passed':bool(np.allclose(a,b,rtol=1e-3,atol=1e-5)),
                     'max_abs_difference':float(np.max(np.abs(a-b)))}
                     for i,(a,b) in enumerate(zip(results[0]['grads'],results[1]['grads']))]
        row={'units':units,'forward_passed':bool(np.allclose(results[0]['output'],results[1]['output'],rtol=1e-4,atol=1e-5)),
             'loss_cpu_mps':[r['loss'] for r in results], 'gradients':differences,
             'seconds_cpu_mps':[r['seconds'] for r in results]}
        rows.append(row)
        print(json.dumps(row),flush=True)
    passed=all(r['forward_passed'] and all(d['passed'] for d in r['gradients']) for r in rows)
    result={'status':'passed-pytorch-cpu-mps-gru' if passed else 'failed-pytorch-cpu-mps-gru',
            'torch':torch.__version__,'rows':rows,'full_model_verified':False}
    (args.output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
