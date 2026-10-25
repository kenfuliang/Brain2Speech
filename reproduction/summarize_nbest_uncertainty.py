"""Append uncertainty to independently verified diagnostics without rewriting them."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from research.reporting.nbest_uncertainty import paired_oracle_interval


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    metrics=json.loads((a.run/'metrics.json').read_text())
    verified=json.loads((a.run/'verification.json').read_text())
    assert verified['status']=='passed-independent-oracle-accounting'
    assert verified['metrics_sha256']==sha(a.run/'metrics.json')
    plan=json.loads((a.run/'uncertainty-plan.json').read_text())
    assert plan['module_sha256']==sha(ROOT/'research/reporting/nbest_uncertainty.py')
    result=dict(scope=metrics['scope'],metrics_sha256=verified['metrics_sha256'],
        verifier_sha256=sha(a.run/'verification.json'),plan_sha256=sha(a.run/'uncertainty-plan.json'),
        curves={n:paired_oracle_interval(curve['trial_accounting'],seed=plan['seed'],repetitions=plan['repetitions'])
                for n,curve in metrics['curves'].items()},
        warning='Paired uncertainty on reference-selected oracle opportunity, not a deployable WER improvement')
    with (a.run/'oracle-uncertainty.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
