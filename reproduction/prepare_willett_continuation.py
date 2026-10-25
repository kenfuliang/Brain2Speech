"""Prepare an isolated E017 child on CPU only; never launches training.

Keep the original identity for the unchanged trainer's resume check and store
new ancestry/approval evidence separately. A prepared child is NOT approved.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from train_pytorch_recipe import ROOT, sha
from willett_fail_stop import require_replay_evidence


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(parent,output,replay,root=ROOT):
    parent=parent.resolve();output=output.resolve();replay=replay.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output==parent or parent in output.parents:
        raise ValueError('Continuation must be outside the preserved parent')
    identity=json.loads((parent/'run-identity.json').read_text())
    for group in ('source_hashes','tfrecord_hashes'):
        for name,expected in identity[group].items():
            if sha(root/name)!=expected:
                raise RuntimeError('Frozen input changed: '+name)
    selected=json.loads((parent/'selected.json').read_text())
    best=Path(selected['checkpoint'])
    if best.is_absolute() or best.name!=str(best):
        raise ValueError('Inherited checkpoint must be a parent-local filename')
    files=['latest.pt','run-identity.json','selected.json',best.name]
    hashes={name:sha(parent/name) for name in files}
    state=torch.load(parent/'latest.pt',map_location='cpu',weights_only=False)
    assert state['seed']==17 and state['next_attempt']==2957 and len(state['history'])==2957
    assert state['best_per']==selected['per']
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert state['optimizer']['param_groups'][0]['iterations']==2957
    for slots in state['optimizer']['state'].values():
        assert slots['step']==2957
        assert torch.isfinite(slots['m']).all() and torch.isfinite(slots['v']).all()
    chosen=torch.load(parent/best,map_location='cpu',weights_only=False)
    assert chosen['seed']==17 and chosen['best_per']==selected['per']
    assert chosen['next_attempt']==selected['attempt']+1
    assert all(torch.isfinite(v).all() for v in chosen['model'].values())
    del state,chosen
    contract={'experiment':'E017','status':'prepared-not-approved',
        'parent':str(parent),'parent_checkpoint_sha256':hashes['latest.pt'],
        'parent_next_attempt':2957,'parent_artifact_sha256':hashes,
        'full_stream_replay_path':str(replay),'full_stream_replay_sha256':sha(replay),
        'full_stream_replay_passed':True,'serial_launch_gate_passed':False,
        'guarded_gpu_resume_gate':'pending-after-BIT-evaluation',
        'safety_wrapper_sha256':sha(root/'reproduction/willett_fail_stop.py'),
        'preparation_runner_sha256':sha(Path(__file__).resolve()),
        'budget':{'seed':17,'next_attempt':2957,'last_attempt':10000,'remaining_attempts':7044},
        'inherited_selection':selected,
        'selection':'Keep inherited minimum PER unless a new 680-trial validation improves it',
        'approval':'No launch; require fresh guarded GPU/resume and serial-resource gates',
        'storage':'Recheck 86.4 GiB potential best checkpoints plus exports/latest and 30 GiB reserve after BIT'}
    require_replay_evidence(contract)
    required=sum((parent/name).stat().st_size for name in files)+30*1024**3
    if shutil.disk_usage(output.parent).free<required:
        raise RuntimeError('Insufficient preparation space plus 30 GiB reserve')
    output.mkdir()  # Deliberately no overwrite or automatic removal on failure.
    save(output/'status.json',{'status':'preparing-no-launch'})
    for name in files:
        shutil.copy2(parent/name,output/name)
        assert sha(output/name)==hashes[name], 'Copied artifact mismatch: '+name
    assert all(sha(parent/name)==expected for name,expected in hashes.items())
    save(output/'continuation-contract.json',contract)
    save(output/'preparation-verification.json',{'passed':True,'cpu_only':True,
        'parent_unchanged':True,'copies_not_hardlinks':all((parent/n).stat().st_ino!=(output/n).stat().st_ino for n in files),
        'copied_sha256':hashes,'inherited_selected_checkpoint_available':True,
        'limitation':'No GPU model/optimizer reload or fresh-process update replay tested here'})
    save(output/'status.json',{'status':'prepared-not-approved','next_attempt':2957,
        'remaining_attempts':7044,'waiting_for':'BIT completion/evaluation, guarded GPU resume, fresh serial-resource checks'})
    return contract


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.parent,args.output,args.replay)
    print('Prepared isolated continuation; training NOT launched',flush=True)


if __name__=='__main__':main()
