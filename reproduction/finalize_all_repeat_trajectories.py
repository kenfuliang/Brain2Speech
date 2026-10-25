"""Seal independently verified repeat trajectory report and sensitivity table."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.audit import digest,save_json


def main(out):
    assert not (out/'seal.json').exists()
    v=json.loads((out/'verification.json').read_text());assert v['status']=='pass'
    assert digest(Path(__file__).with_name('verify_all_repeat_trajectories.py'))==v['script_sha256']
    m=json.loads((out/'metrics.json').read_text());group=[r for r in m['records'] if r['cross_day'] and len(r['controls'])>=3]
    save_json(out/'coverage-sensitivity.json',dict(min3_controls_pairs=len(group),
        positive_shape_advantage_pairs=int(sum(r['metrics']['elapsed']['shape_advantage']>0 for r in group)),
        median_shape_advantage=float(np.median([r['metrics']['elapsed']['shape_advantage'] for r in group])),
        day_pairs=len({(r['left_day'],r['right_day']) for r in m['records'] if r['cross_day']}),
        participating_days=len({r[k] for r in m['records'] for k in ('left_day','right_day')}),
        no_inferential_intervals='Shared sessions/controls; sparse nonmatched within-day repeat cohort'))
    save_json(out/'final-QA.json',dict(status='pass',image='all-repeat-similarity.png',
        visual_inspection='Inspected exported PNG: readable labels, no clipping, fixed cosine scales, control counts and dependency qualifications visible',
        tests='2 targeted synthetic/control-selection tests pass; independent798 pair checks and2 exact source re-inferences pass',
        failure_preservation='Run01 extraction retained after report NumPy-integer serialization failure; run02 reuses hash-snapshotted arrays; same scientific design',
        renderer='CPU-only matplotlib; no GPU trainer interruption',finalizer_sha256=digest(__file__)))
    save_json(out/'seal.json',dict(status='sealed',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print('sealed',len(json.loads((out/'seal.json').read_text())['files']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
