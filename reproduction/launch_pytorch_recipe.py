"""Detached local launch, fresh artifacts only; no cloud or paid compute."""
import argparse
import json
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--gate',type=Path,required=True); p.add_argument('--seed',type=int,default=17); a=p.parse_args()
    root=Path(__file__).absolute().parent.parent
    out=a.output.absolute()
    if out.exists(): raise FileExistsError(out)
    log=out.with_suffix('.launch.log'); record=out.with_suffix('.launch.json')
    if record.exists(): raise FileExistsError(record)
    command=[str(root/'.venv-v1/bin/python'),'-u',str(root/'reproduction/train_pytorch_recipe.py'),'--output',str(out),'--seed',str(a.seed),'--require-gate',str(a.gate.absolute())]
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    value=dict(pid=proc.pid,command=command,log=str(log),output=str(out),seed=a.seed,scope='Full local E008 seed, not v1; WER evaluation follows training')
    record.write_text(json.dumps(value,indent=2)+'\n'); print(json.dumps(value))

if __name__=='__main__': main()
