"""Independent fit/accounting/trajectory checks; no neural inference or updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.audit import digest, save_json


def main(run):
    protocol=json.loads((run/'protocol.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
    for path,h in protocol['input_hashes'].items():assert digest(path)==h
    assert digest(Path(__file__).with_name('quantify_trajectory_correction.py'))==protocol['script_sha256']
    cohort_path=next(Path(p) for p in protocol['input_hashes'] if p.endswith('e018-neural-drift-20261011-01/cohort.json'))
    summary_path=cohort_path.with_name('summaries.npz')
    rows=json.loads(cohort_path.read_text());lookup={r['id']:r for r in rows};index={r['id']:i for i,r in enumerate(rows)}
    fit_ids={r['id'] for rs in protocol['fit'].values() for r in rs};fit_text={lookup[i]['text'] for i in fit_ids}
    comparisons=protocol['evaluation'];positives={p['prompt'] for p in comparisons}
    eval_ids={p[k] for p in comparisons for k in ('left','right')}|{i for p in comparisons for i in p['controls']}
    assert not fit_ids&eval_ids and not fit_text&{lookup[i]['text'] for i in eval_ids}
    assert len(fit_ids)==40 and len(eval_ids)==153 and len(comparisons)==10
    for d,fit in protocol['fit'].items():
        assert [r['id'] for r in fit]==[r['id'] for r in rows if r['session']==d][:20]
        assert max(r['index'] for r in fit)<min(lookup[i]['index'] for i in eval_ids if lookup[i]['session']==d)
    arrays={}
    for i in eval_ids:
        r=lookup[i];path=run/f"{r['session']}-{r['index']:04d}.npz"
        with np.load(path) as data:
            h=data['hidden'].copy();assert h.shape==((r['bins']-32)//4+1,1024) and np.isfinite(h).all()
            assert np.array_equal(data['seconds'],(32+4*np.arange(len(h)))*.02)
        arrays[i]=h
    with np.load(summary_path) as s,np.load(run/'correction.npz') as saved:
        for n in (10,20):
            estimates=[]
            for day in protocol['days']:
                ids=[index[r['id']] for r in protocol['fit'][day][:n]]
                mu=sum(s['latent_mean'][i] for i in ids)/n
                second=sum(s['latent_variance'][i]+s['latent_mean'][i]**2 for i in ids)/n
                sd=np.sqrt(np.maximum(second-mu**2,0));estimates.append((mu,sd))
            ma,sa=estimates[0];mb,sb=estimates[1];gain=np.clip(sa/np.maximum(sb,1e-6),.5,2.)
            for key,value in [('reference_mean',ma),('target_mean',mb),('gain',gain)]:
                np.testing.assert_allclose(saved[f'n{n}_{key}'],value,atol=1e-12,rtol=1e-12)
            for mode in ('raw','offset','offset_gain'):
                records=metrics['results'][str(n)][mode]['records']
                for pair,record in zip(comparisons,records):
                    assert pair['prompt']==record['prompt']
                    target=lookup[pair['right']]
                    candidates=[r for r in rows if r['session']==protocol['days'][1] and r['index']>=20 and r['block']==target['block'] and r['text'] not in fit_text|positives]
                    import math
                    candidates.sort(key=lambda r:(abs(math.log(r['bins']/target['bins'])),r['index']))
                    assert pair['controls']==[r['id'] for r in candidates[:20]]
                    for right,stored in zip([pair['right']]+pair['controls'],[record['positive']]+record['negatives']):
                        a=arrays[pair['left']];b=arrays[right]
                        if mode!='raw':b=(b-mb)*(gain if mode=='offset_gain' else 1)+ma
                        length=min(len(a),len(b));a=a[:length];b=b[:length]
                        d=(a-b).ravel();rms=np.linalg.norm(d)/np.sqrt(d.size)
                        ac=a-a.sum(0)/length;bc=b-b.sum(0)/length
                        shape=np.dot(ac.ravel(),bc.ravel())/(np.linalg.norm(ac)*np.linalg.norm(bc))
                        assert np.isclose(stored['RMS'],rms,atol=1e-12)
                        assert np.isclose(stored['shape_cosine'],shape,atol=1e-12)
                        centroid=np.mean((a.mean(0)-b.mean(0))**2)
                        assert np.isclose(rms**2,centroid+np.mean((ac-bc)**2),atol=1e-12)
    loo={}
    raw=metrics['results']['20']['raw']['records'];gain_records=metrics['results']['20']['offset_gain']['records']
    for key in ('RMS_advantage','shape_advantage','RMS_rank'):
        values=[]
        for dropped in range(10):
            chosen=[i for i in range(10) if i!=dropped]
            values.append(float(np.median([gain_records[i][key] for i in chosen])-np.median([raw[i][key] for i in chosen])))
        loo[key]=dict(median_change_range=[min(values),max(values)],interpretation='Leave-one-prompt-out sensitivity, not a confidence interval')
    output=dict(status='pass',fit_evaluation_disjoint=True,prompt_disjoint=True,preceding_prefixes=True,
                independent_moments=True,independent_full_space_metrics=True,distance_decomposition=True,
                deterministic_duration_control_selection=True,leave_one_prompt_out=loo,
                checks='All2prefixes/3modes/10positive and199negative instances checked; no bins treated as independent observations',
                script_sha256=digest(__file__),protocol_sha256=digest(run/'protocol.json'),metrics_sha256=digest(run/'metrics.json'))
    assert not (run/'verification.json').exists();save_json(run/'verification.json',output);print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
