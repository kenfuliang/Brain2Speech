"""Full-size deterministic port gate; CPU CTC explicitly retains autograd to MPS."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from pytorch_baseline import PublishedGRU,KerasAdam

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=('cpu','mps'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).absolute().parent.parent
    if a.output.exists(): raise FileExistsError(a.output)
    if shutil.disk_usage(a.output.parent).free<30*1024**3: raise RuntimeError('Storage guard')
    a.output.mkdir()
    torch.set_num_threads(8)
    f=np.load(root/'reproduction/results/retraining/pytorch-fixture-20261007-01.npz')
    model=PublishedGRU(); model.load_tf(f); model.to(a.device).eval()
    opt=KerasAdam(model.parameters())
    x=torch.tensor(f['input'],device=a.device)
    targets=torch.tensor(f['labels'].astype('int64'))-1
    lengths=torch.tensor(((f['lengths']-32)//4+1).astype('int64'))
    label_lengths=torch.tensor(f['label_lengths'].astype('int64'))
    start=time.monotonic()
    y=model(x)
    loss=torch.nn.functional.ctc_loss(y.cpu().log_softmax(-1).transpose(0,1),targets,lengths,label_lengths,blank=40,reduction='none').mean().to(a.device)+model.regularization()
    loss.backward()
    arrays={'logits':y.detach().cpu().numpy(),'loss':np.asarray(float(loss.detach().cpu()))}
    arrays.update({f'gradient_{i:03d}':v for i,v in enumerate(model.tf_arrays(True))})
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
    opt.step()
    arrays.update({f'weight_{i:03d}':v for i,v in enumerate(model.tf_arrays())})
    ref=np.load(root/'reproduction/results/retraining/native-validation-20261007-cpu-01/values.npz')
    comparisons=[]
    for k,v in arrays.items():
        rtol=1e-3 if k.startswith('gradient') else 1e-4
        comparisons.append(dict(name=k,passed=bool(np.allclose(v,ref[k],rtol=rtol,atol=1e-5)),max_abs_difference=float(np.max(np.abs(v-ref[k])))))
    passed=all(c['passed'] for c in comparisons)
    result=dict(status='passed' if passed else 'failed-parity',device=a.device,torch=torch.__version__,loss=float(loss.detach().cpu()),gradient_norm=float(norm.detach().cpu()),seconds=time.monotonic()-start,parameter_count=sum(p.numel() for p in model.parameters()),comparisons=comparisons,
                ctc_device='cpu-explicit-differentiable-transfer',dropout_noise_enabled=False,full_training_started=False,wer=None,
                data_sha256=hashlib.sha256(f['raw'].tobytes()).hexdigest())
    (a.output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}),flush=True)
    return 0 if passed else 1

if __name__=='__main__': raise SystemExit(main())
