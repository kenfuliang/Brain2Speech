"""Versioned safety wrapper for the unchanged E008 trainer; no launch on import.

Use only in a separately prepared, provenance-pinned continuation directory.
Normal finite arithmetic is unchanged; unsafe steps fail before sealing outputs.
"""
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

import torch
import train_pytorch_recipe as recipe

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def require_serial_resources():
    """Fresh fail-closed inventory; a saved approval boolean is not evidence."""
    if shutil.disk_usage(Path(__file__).resolve().parents[1]).free < 30*1024**3:
        raise RuntimeError('Willett launch blocked: less than 30 GiB free')
    processes=subprocess.check_output(['ps','-ax','-o','pid=,command='],text=True).splitlines()
    for line in processes:
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():
            continue
        command=fields[1]
        executable=Path(shlex.split(command)[0]).name.lower()
        if not executable.startswith(('python','pypy')):
            continue  # A launcher shell mentioning a script is not a trainer.
        if any(name in command for name in ('train_bit_full.py','train_bit_ctc_pilot.py',
                'train_pytorch_recipe.py','willett_fail_stop.py','research.experiment.pipeline --stage')):
            raise RuntimeError('Willett launch blocked: competing trainer/controller PID '+fields[0])
    from research.experiment.evaluator import IMAGE
    resident=subprocess.check_output(['docker','--context','colima-speechbci','ps',
        '--filter','ancestor='+IMAGE,'--format','{{.ID}}'],text=True).strip()
    if resident:
        raise RuntimeError('Willett launch blocked: resident fixed decoder '+resident)


def require_replay_evidence(contract):
    path=Path(contract['full_stream_replay_path']).resolve()
    assert recipe.sha(path)==contract['full_stream_replay_sha256'], 'Replay evidence changed'
    evidence=json.loads(path.read_text())
    assert evidence['status']=='passed-full-cpu-stream-replay'
    assert evidence['parent_unchanged'] and evidence['optimizer_and_rng_payload_verified']
    assert evidence['checkpoint_sha256']==contract['parent_checkpoint_sha256']
    assert evidence['records']==contract['parent_next_attempt']
    parent=Path(contract['parent']).resolve()
    assert recipe.sha(parent/'latest.pt')==contract['parent_checkpoint_sha256'], 'Parent checkpoint changed'


def require_finite(value,description):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError('Willett fail-stop: nonfinite '+description)
    return value


def install_guards():
    if getattr(recipe,'_finite_guards_installed',False):
        return
    original_ctc=recipe.ctc
    original_clip=torch.nn.utils.clip_grad_norm_
    original_step=recipe.KerasAdam.step

    def checked_ctc(*args,**kwargs):
        return require_finite(original_ctc(*args,**kwargs),'CTC loss')

    def checked_clip(*args,**kwargs):
        kwargs['error_if_nonfinite']=True
        return original_clip(*args,**kwargs)

    def checked_step(optimizer,*args,**kwargs):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                require_finite(parameter,'pre-update weight')
                if parameter.grad is not None:
                    require_finite(parameter.grad,'gradient')
        result=original_step(optimizer,*args,**kwargs)
        for group in optimizer.param_groups:
            for parameter in group['params']:
                require_finite(parameter,'post-update weight')
                for key in ('m','v'):
                    require_finite(optimizer.state[parameter][key],'Adam '+key)
        return result

    recipe.ctc=checked_ctc
    torch.nn.utils.clip_grad_norm_=checked_clip
    recipe.KerasAdam.step=checked_step
    recipe._finite_guards_installed=True


def main():
    # The launcher must pin this wrapper and prepare the child independently.
    index=sys.argv.index('--output')
    output=Path(sys.argv[index+1]).resolve()
    assert (output/'continuation-contract.json').exists(), 'Missing frozen child continuation contract'
    contract=json.loads((output/'continuation-contract.json').read_text())
    assert recipe.sha(Path(__file__).resolve())==contract['safety_wrapper_sha256']
    assert str(output)!=contract['parent'], 'Never wrap/rewrite the preserved parent run'
    assert contract['full_stream_replay_passed'] and contract['serial_launch_gate_passed']
    require_replay_evidence(contract)
    require_serial_resources()
    install_guards()
    try:
        recipe.main()
    except Exception as error:
        recipe.write(output/'safety-failure.json',{'status':'failed-stop','error':str(error),
            'recovery':'Restore last sealed child checkpoint; failed/interrupted steps are not completion evidence'})
        raise


if __name__=='__main__':main()
