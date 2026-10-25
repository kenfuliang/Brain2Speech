"""Alternate aggregation/whole-session sensitivity and final evidence seal."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.audit import digest,save_json


def main(out):
    assert not (out/'seal.json').exists()
    p=json.loads((out/'protocol.json').read_text());run=Path(p['parent'])
    assert digest(run/'protocol.json')==p['parent_protocol_sha256']
    assert digest(Path(__file__).with_name('summarize_causal_recipe_evaluation.py'))==p['script_sha256']
    m=json.loads((out/'metrics.json').read_text());v=json.loads((out/'verification.json').read_text());assert v['status']=='pass'
    for path,h in m['source_hashes'].items():assert digest(path)==h
    import re
    assert json.loads((run/'status.json').read_text())['status']=='decoded-independent-comparison-audit-pending'
    budgets=[0,5,10,25,50,100,'full'];day_sensitivity={}
    for budget in budgets:
        for mode in (['normalization'] if budget==0 else ['normalization','adapter']):
            raw=[];ans=[]
            for d in m['calibration_curves']['raw']:
                a=json.loads((run/'raw'/d/str(budget)/mode/'decoded.json').read_text());b=json.loads((run/'anscombe'/d/str(budget)/mode/'decoded.json').read_text())
                assert all(x['id']==y['id'] and x['text']==y['text'] and x['reference_phones']==y['reference_phones'] for x,y in zip(a,b))
                for f in ('wer','per','cer'):assert all(x[f]['reference_units']==y[f]['reference_units'] for x,y in zip(a,b))
                raw.append(a);ans.append(b)
            rng=np.random.default_rng(101);indices=rng.integers(0,5,size=(2000,5));scope={}
            for f in ('wer','per','cer'):
                ar=np.array([sum(r[f]['errors'] for r in rows) for rows in raw]);br=np.array([sum(r[f]['errors'] for r in rows) for rows in ans]);n=np.array([sum(r[f]['reference_units'] for r in rows) for rows in raw])
                for arm,e in [('raw',ar),('anscombe',br)]:np.testing.assert_allclose(m['conditions'][f'{arm}/{budget}/{mode}'][f]['corpus'],e.sum()/n.sum(),rtol=0,atol=1e-15)
                delta=(br.sum()-ar.sum())/n.sum();np.testing.assert_allclose(delta,m['paired_deltas'][f'{budget}/{mode}'][f]['anscombe_minus_raw'],rtol=0,atol=1e-15)
                draws=(br[indices].sum(1)-ar[indices].sum(1))/n[indices].sum(1)
                scope[f]=dict(anscombe_minus_raw=float(delta),ci95=np.quantile(draws,[.025,.975]).tolist())
            day_sensitivity[f'{budget}/{mode}']=scope
    for arm,days in m['calibration_curves'].items():
        for d,curve in days.items():
            eligible=[r['labels'] for r in curve if r['wer']<=.25]
            assert m['labels_to_25percent_WER'][arm][d]==(eligible[0] if eligible else 'not reached')
    save_json(out/'whole-session-sensitivity.json',dict(method='2000 paired whole-session resamples,seed101; retain all40trials within selected days, no trial independence assumption',results=day_sensitivity,limitation='Only5historically exposed days/onepairedseed; conditional, no multiplicity correction'))
    save_json(out/'final-QA.json',dict(status='pass',aggregate_point_estimates='Independent day-sum reconstruction',matched_references='Exact all conditions',calibration_crossings='Measured budgets only, checked',
        tests='3new summary tests pass; earlier3evaluation tests/export5200phone/text15600edit checks pass',
        image='raw-anscombe-calibration-v02.png',image_QA='Inspected PNG after display-only repair: both-arm limits prevent clipping; legends avoid markers; sample/coverage caveats visible',
        owned_exit='OriginalPID47552andcontainer30b4bde18147absent in read-only process/Docker inventory before audit',script_sha256=digest(__file__)))
    save_json(out/'seal.json',dict(status='sealed',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print('sealed',len(json.loads((out/'seal.json').read_text())['files']),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);main(a.parse_args().output.resolve())
