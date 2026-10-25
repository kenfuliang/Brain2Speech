"""Paired hierarchical uncertainty for oracle diagnostics, not deployed gains."""
import numpy as np


def paired_oracle_interval(accounting,seed=101,repetitions=2000):
    if not accounting or repetitions<100: raise ValueError('Require trials and at least 100 resamples')
    ids=[r['id'] for r in accounting]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate trial identities')
    days=sorted({r['session'] for r in accounting})
    groups={day:[] for day in days}
    for r in accounting:
        t,o=r['top1'],r['oracle']
        if t['reference_units']!=o['reference_units'] or t['reference_units']<=0:
            raise ValueError('Mismatched/empty paired denominator')
        if not 0<=o['errors']<=t['errors']: raise ValueError('Invalid oracle errors')
        groups[r['session']].append([t['errors'],o['errors'],t['reference_units']])
    groups={day:np.asarray(values,dtype=np.float64) for day,values in groups.items()}
    totals=sum((g.sum(0) for g in groups.values()),np.zeros(3))
    point=[totals[0]/totals[2],totals[1]/totals[2],(totals[0]-totals[1])/totals[2]]
    if len(days)<3:
        return dict(trials=len(accounting),sessions=len(days),point=dict(zip(['top1_wer','oracle_wer','oracle_gap'],point)),
                    ci95=None,reason='Fewer than three sessions; no inferential interval reported',diagnostic_only=True)
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        total=np.zeros(3)
        for day in rng.choice(days,len(days),replace=True):
            group=groups[day];total+=group[rng.integers(0,len(group),len(group))].sum(0)
        samples.append([total[0]/total[2],total[1]/total[2],(total[0]-total[1])/total[2]])
    intervals=np.quantile(samples,[.025,.975],axis=0)
    return dict(trials=len(accounting),sessions=len(days),seed=seed,repetitions=repetitions,
                point=dict(zip(['top1_wer','oracle_wer','oracle_gap'],point)),
                ci95={name:intervals[:,i].tolist() for i,name in enumerate(['top1_wer','oracle_wer','oracle_gap'])},
                method='Paired session-then-trial bootstrap; identical draws for top-1 and oracle',
                diagnostic_only=True,
                warning='One participant, few sessions; oracle uses labels and is not deployable improvement or seed dispersion')
