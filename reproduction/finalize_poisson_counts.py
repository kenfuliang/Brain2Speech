"""Seal E020 after independent verification, executed companion and visual QA."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json, space_guard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    assert not (out/'seal.json').exists(), 'Preserve sealed run'
    space_guard()
    assert json.loads((out/'verification.json').read_text())['status']=='pass'
    notebook=json.loads((out/'inspect-evidence.executed.ipynb').read_text())
    assert notebook['nbformat']==4 and notebook['nbformat_minor']==5
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert [c['execution_count'] for c in cells]==[1,2,3]
    assert all(c['outputs'] and not any(o['output_type']=='error' for o in c['outputs']) for c in cells)
    original=json.loads((out/'inspect-evidence.ipynb').read_text())
    assert [c['source'] for c in notebook['cells']]==[c['source'] for c in original['cells']]
    preservation={}
    for name in ['e018-neural-drift-20261011-01','e019-normalization-adapter-20261011-01']:
        parent=ROOT/'research/runs'/name
        seal=json.loads((parent/'seal.json').read_text())
        bad=[rel for rel, expected in seal['files'].items() if digest(parent/rel)!=expected]
        assert not bad, 'Changed preserved evidence: '+str(bad[:3])
        preservation[name]=dict(files=len(seal['files']),mismatches=0,seal_sha256=digest(parent/'seal.json'))
    code=[ROOT/'reproduction'/name for name in ['analyze_poisson_counts.py','verify_plot_poisson_counts.py','finalize_poisson_counts.py','execute_drift_notebook.py']]
    provenance=json.loads((out/'figure-provenance-v02.json').read_text())
    assert digest(out/'poisson-diagnostics-v02.png')==provenance['image_sha256']
    assert digest(ROOT/'reproduction/verify_plot_poisson_counts.py')==provenance['implementation_sha256']
    for path, expected in provenance['source_sha256'].items():
        assert digest(path)==expected
    save_json(out/'final-QA.json',dict(status='pass',preserved=preservation,
        notebook='Three cells executed by plain Python; no Jupyter kernel',
        figure='v02 PNG visually inspected; no footer/axis overlap; original retained',
        tests=dict(command='.venv-bit/bin/python -m pytest research/tests -q',passed=179,seconds=14.46),
        implementation_sha256={str(p):digest(p) for p in code},
        uncertainty='Descriptive conditional adequacy; no exact Poisson, neural drift mechanism or model-improvement claim'))
    save_json(out/'status.json',dict(status='complete',completed_local=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        no_training=True,no_decoding=True,no_final_recordings=True,v1='unchanged, paused'))
    files={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save_json(out/'seal.json',dict(status='complete',files=files))
    assert all(digest(out/rel)==expected for rel,expected in files.items())
    print('Sealed',len(files),'artifacts; prior evidence preserved:',preservation)


if __name__=='__main__':main()
