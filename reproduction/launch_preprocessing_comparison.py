"""One-shot serial E021 training/calibration/decoding; not a recurring monitor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os
import fcntl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.audit import save_json,digest,space_guard
from export_willett_final import require_idle
from research.experiment.evaluator import IMAGE,IMAGE_ID


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    lock=(run/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require_idle();space_guard()
    assert json.loads((run/'pilot.json').read_text())['status']=='pass'
    assert json.loads((run/'resume-gate.json').read_text())['status']=='pass'
    scripts=['run_preprocessing_comparison.py','compare_preprocessing_results.py','launch_preprocessing_comparison.py']
    contract=dict(implementation={name:digest(Path(__file__).with_name(name))for name in scripts},
        derived_inputs={name:digest(run/name)for name in ['protocol.json','splits.json','raw-normalization.json','sqrt-normalization.json','anscombe-normalization.json','pilot.json','resume-gate.json']})
    contract['LM_files']={str(Path(__file__).resolve().parents[1]/'workspace/languageModel'/name):
        digest(Path(__file__).resolve().parents[1]/'workspace/languageModel'/name)for name in ['TLG.fst','words.txt']}
    observed=subprocess.check_output(['docker','--context','colima-speechbci','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    assert observed==IMAGE_ID;contract['decoder_image_id']=observed
    prior=run/'execution-contract.json'
    if prior.exists():assert json.loads(prior.read_text())==contract
    else:save_json(prior,contract)
    save_json(run/'launch.json',dict(pid=os.getpid(),mode='serial one-shot; no automation or goal/v1 changes'))
    for stage in ['train','calibrate','decode']:
        assert all(digest(Path(__file__).with_name(n))==h for n,h in contract['implementation'].items())
        assert all(digest(run/n)==h for n,h in contract['derived_inputs'].items())
        assert all(digest(p)==h for p,h in contract['LM_files'].items())
        save_json(run/'controller-status.json',dict(status='running',stage=stage,pid=os.getpid()))
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_preprocessing_comparison.py')),
            '--output',str(run),'--stage',stage],check=True)
        assert all(digest(run/n)==h for n,h in contract['derived_inputs'].items())
        assert json.loads((run/'status.json').read_text())['status']!='stopped-resumable'
    subprocess.run([sys.executable,str(Path(__file__).with_name('compare_preprocessing_results.py')),'--run',str(run)],check=True)
    save_json(run/'controller-status.json',dict(status='comparison-collected-report-reconciliation-pending',pid=os.getpid()))


if __name__=='__main__':main()
