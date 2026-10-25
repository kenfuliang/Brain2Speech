"""Audit and seal the user-stopped E021 run; delete nothing, resume nothing."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    assert not (run/'seal-stopped.json').exists()
    status=json.loads((run/'status.json').read_text());assert status['status']=='stopped-resumable'
    launch=json.loads((run/'launch.json').read_text())
    live=subprocess.run(['ps','-p',str(launch['pid']),'-o','command='],capture_output=True,text=True)
    assert 'launch_preprocessing_comparison.py'not in live.stdout
    commands=subprocess.check_output(['ps','-axo','command='],text=True)
    assert not any('run_preprocessing_comparison.py' in line and str(run)in line for line in commands.splitlines())
    payload=torch.load(run/'raw/latest.pt',map_location='cpu',weights_only=False)
    assert payload['step']==status['step']==len(payload['history'])
    assert [r['step']for r in payload['history']]==list(range(1,payload['step']+1))
    assert all(np.isfinite(r['ctc'])for r in payload['history'])
    assert all(torch.isfinite(v).all()for v in payload['model'].values())
    for state in payload['optimizer']['state'].values():
        assert all(torch.isfinite(v).all()for v in state.values()if isinstance(v,torch.Tensor))
    assert {'rng','torch_rng','mps_rng','optimizer','selected'}<=payload.keys()
    selected=run/'raw'/payload['selected'];assert selected.exists()
    assert not (run/'sqrt/progress.json').exists() and not (run/'anscombe/progress.json').exists()
    original=json.loads((run/'controller-status.json').read_text())
    record=dict(status='user-stopped-preserved',local_time=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        stop_reason='User requested recipe-first raw recovery before transformed-count experiments',
        raw_updates=payload['step'],raw_selected=payload['selected'],selection_macro_PER=payload['best'],
        first200_mean_CTC=float(np.mean([r['ctc']for r in payload['history'][:200]])),
        last200_mean_CTC=float(np.mean([r['ctc']for r in payload['history'][-200:]])),
        latest_sha256=digest(run/'raw/latest.pt'),selected_sha256=digest(selected),
        history_and_model_optimizer_finite=True,RNG_preserved=True,controller_exited=True,
        controller_status_before_reconciliation=original,
        transformed_production_runs='Never started; technical pilot artifacts preserved',
        WER='Not measured',later_day_PER='Not measured',v1='Unchanged,paused',
        no_deletions=True,recovery='Any replacement is a fresh named study; no automatic E021 restart')
    save_json(run/'stop-preservation.json',record)
    save_json(run/'controller-status.json',dict(status='user-stopped-preserved',stage='train',raw_step=payload['step']))
    files={str(p.relative_to(run)):digest(p)for p in sorted(run.rglob('*'))if p.is_file()}
    save_json(run/'seal-stopped.json',dict(status='user-stopped-preserved',files=files))
    assert all(digest(run/p)==h for p,h in files.items())
    print(json.dumps(record,indent=2));print('Preserved and sealed',len(files),'artifacts')


if __name__=='__main__':main()
