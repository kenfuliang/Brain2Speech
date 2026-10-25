"""CPU-only full TF data-stream replay audit; no neural model training or GPU.

Validate preserved E008 payload and actual stream position before serial resume.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from train_pytorch_recipe import ROOT, RecipeData, data_hash, sha, shutil_free


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parent=args.parent.resolve();out=args.output.resolve()
    assert shutil_free(ROOT)>30*1024**3
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=parent/'latest.pt';checkpoint_hash=sha(checkpoint)
    identity=json.loads((parent/'run-identity.json').read_text())
    assert all(sha(ROOT/k)==h for k,h in identity['source_hashes'].items())
    assert all(sha(ROOT/k)==h for k,h in identity['tfrecord_hashes'].items())
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['seed']==17 and state['next_attempt']==2957
    assert len(state['history'])==2957
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    group=state['optimizer']['param_groups'][0]
    assert group['iterations']==2957 and abs(group['lr']-.02*(1-2956/10000))<1e-12
    assert state['torch_rng'].dtype==torch.uint8 and state['mps_rng'].dtype==torch.uint8
    for payload in state['optimizer']['state'].values():
        assert payload['step']==2957
        assert torch.isfinite(payload['m']).all() and torch.isfinite(payload['v']).all()
    journal=[json.loads(line) for line in (parent/'training.jsonl').read_text().splitlines()]
    assert len(journal)==2957 and [r['attempt'] for r in journal]==list(range(2957))
    assert all(r['update_applied'] for r in journal)
    assert [(r['day'],r['data_hash']) for r in journal]==[(r['day'],r['data_hash']) for r in state['history']]
    sampler=np.random.RandomState(17)
    probabilities=identity['configuration']['dataset']['datasetProbability']
    assert [int(np.argwhere(sampler.multinomial(1,probabilities))[0][0]) for _ in journal]==[r['day'] for r in journal]
    saved_rng=state['numpy_rng'];current_rng=sampler.get_state()
    assert saved_rng[0]==current_rng[0] and np.array_equal(saved_rng[1],current_rng[1]) and saved_rng[2:]==current_rng[2:]
    save(out/'contract.json',{'parent':str(parent),'checkpoint_sha256':checkpoint_hash,
        'scope':'CPU-only original TF data-service replay; no neural forward/backward/optimizer/GPU work',
        'records':2957,'source_hashes_verified':True,'tfrecord_hashes_verified':True,
        'runner_sha256':sha(Path(__file__).resolve()),'required_next_attempt':2957})
    service=RecipeData(out,17);started=time.monotonic()
    try:
        service.call('initial')
        for i,row in enumerate(state['history'],1):
            value=service.call('train',day=row['day'])
            if data_hash(value)!=row['data_hash']:
                raise RuntimeError(f'TF stream mismatch at historical attempt{i-1}; forbid claimed exact resume')
            if i%100==0 or i==len(state['history']):
                progress={'status':'replaying','verified':i,'total':2957,'elapsed_seconds':time.monotonic()-started}
                save(out/'status.json',progress);print(json.dumps(progress),flush=True)
        assert sha(checkpoint)==checkpoint_hash
        result={'status':'passed-full-cpu-stream-replay','records':2957,'checkpoint_sha256':checkpoint_hash,
                'parent_unchanged':True,'optimizer_and_rng_payload_verified':True,
                'elapsed_seconds':time.monotonic()-started,
                'limitation':'No GPU model/optimizer reload exercised; validate it after BIT before training resumes'}
        save(out/'result.json',result);save(out/'status.json',result)
    except Exception as error:
        save(out/'failure.json',{'error':str(error),'status':'failed-replay'})
        raise
    finally:
        service.close()


if __name__=='__main__':main()
