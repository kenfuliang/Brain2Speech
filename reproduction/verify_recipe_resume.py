"""Independent fresh-process continuation gate before full E008 collection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).absolute().parent.parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('--reference',type=Path,required=True); p.add_argument('--resumed',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    ref=torch.load(a.reference/'latest.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(a.resumed/'latest.pt',map_location='cpu',weights_only=False)
    assert ref['history']==resumed['history']
    assert ref['next_attempt']==resumed['next_attempt']==3
    assert ref['best_per']==resumed['best_per']
    weight_rows=[]
    for k,v in ref['model'].items():
        other=resumed['model'][k]
        weight_rows.append(dict(name=k,exact=bool(torch.equal(v,other)),max_abs_difference=float((v-other).abs().max())))
    assert all(r['exact'] for r in weight_rows), 'Restored continuation weights differ'
    assert ref['optimizer']['param_groups']==resumed['optimizer']['param_groups']
    for key,slots in ref['optimizer']['state'].items():
        for name,value in slots.items():
            other=resumed['optimizer']['state'][key][name]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
    np.testing.assert_array_equal(np.load(a.reference/'final-probe-logits.npy'),np.load(a.resumed/'final-probe-logits.npy'))
    assert torch.equal(ref['torch_rng'],resumed['torch_rng'])
    assert torch.equal(ref['mps_rng'],resumed['mps_rng'])
    for x,y in zip(ref['numpy_rng'],resumed['numpy_rng']): np.testing.assert_array_equal(x,y)
    validation=json.loads((a.reference/'validation-00000.json').read_text())
    assert validation['trials']==680 and sum(int(v) for k,v in validation['per_day_trials'].items() if int(k)>=4)==600
    paths=[ROOT/'reproduction/train_pytorch_recipe.py',ROOT/'reproduction/pytorch_baseline.py',ROOT/'reproduction/tf_recipe_service.py',ROOT/'workspace/derived/rnns/baselineRelease/args.yaml']
    identity=json.loads((a.reference/'run-identity.json').read_text())
    result=dict(status='passed-recipe-and-fresh-resume',reference=str(a.reference),resumed=str(a.resumed),
                model_and_optimizer_exact=True,data_stream_and_rng_exact=True,validation_trials=680,benchmark_trials=600,
                weight_comparisons=weight_rows,source_hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
                reference_run_source_hashes=identity['source_hashes'],
                limitations=['Three-update port validation, not convergence or WER','PyTorch dropout random draws differ from TensorFlow; distribution and time-lock semantics retained','CPU recurrent fallback not enabled; explicit CPU CTC is the tested route'])
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(result['status'])

if __name__=='__main__': main()
