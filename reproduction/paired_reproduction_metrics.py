"""Prospective E017 artifact-relative WER comparison; no model execution.

Caller must supply sealed, same-normalization benchmark rows. This function
cannot establish matching architecture/training or population equivalence.
"""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.metrics import edits


def paired_wer(candidate,baseline,*,repetitions=2000,seed=101,band=.01,
               published_target=.188,tokenization='standard-whitespace'):
    if tokenization not in ('standard-whitespace','legacy-literal-space'):
        raise ValueError('Unknown word tokenization')
    if not candidate or len(candidate)!=len(baseline):
        raise ValueError('Require nonempty equal-sized aligned cohorts')
    if repetitions<1 or band<=0:
        raise ValueError('Invalid prospective comparison settings')
    def index(rows):
        result={}
        for row in rows:
            if row['trial_id'] in result:
                raise ValueError('Duplicate trial identity')
            result[row['trial_id']]=row
        return result
    candidate_map=index(candidate);baseline_map=index(baseline)
    if candidate_map.keys()!=baseline_map.keys():
        raise ValueError('Candidate/baseline trial identities differ')
    groups=defaultdict(list);components=[]
    tokenize=(lambda text:text.split()) if tokenization=='standard-whitespace' else (lambda text:text.split(' '))
    for row in candidate:
        other=baseline_map[row['trial_id']]
        if row['session']!=other['session'] or row['reference']!=other['reference']:
            raise ValueError('Session/reference alignment differs: '+row['trial_id'])
        reference=tokenize(row['reference'])
        if not reference:
            raise ValueError('Empty reference is outside this speech benchmark')
        a=edits(reference,tokenize(row['transcript']))
        b=edits(reference,tokenize(other['transcript']))
        groups[row['session']].append((a['errors'],b['errors'],len(reference)))
        components.append((a,b))
    days=sorted(groups);units=sum(v[2] for rows in groups.values() for v in rows)
    candidate_errors=sum(v[0] for rows in groups.values() for v in rows)
    baseline_errors=sum(v[1] for rows in groups.values() for v in rows)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repetitions):
        a=b=n=0
        for day in rng.choice(days,len(days)):
            group=groups[day]
            for i in rng.integers(0,len(group),len(group)):
                ca,ba,nu=group[i];a+=ca;b+=ba;n+=nu
        draws.append((a-b)/n)
    interval=np.quantile(draws,[.025,.975]).tolist()
    equivalent=interval[0]>=-band and interval[1]<=band
    return {'trials':len(candidate),'sessions':len(days),'tokenization':tokenization,
        'reference_words':units,'candidate_wer':candidate_errors/units,
        'baseline_wer':baseline_errors/units,'paired_gap_candidate_minus_baseline':(candidate_errors-baseline_errors)/units,
        'paired_gap_ci95':interval,'artifact_relative_equivalence':equivalent,
        'published_target':published_target,'candidate_minus_published_target':candidate_errors/units-published_target,
        'inside_exploratory_point_band':abs(candidate_errors/units-published_target)<=band,
        'agreement_band':[-band,band],'bootstrap_repetitions':repetitions,'bootstrap_seed':seed,
        'error_components':{name:{key:sum(pair[i][key] for pair in components)
            for key in ('errors','substitutions','deletions','insertions')}
            for i,name in enumerate(('candidate','baseline'))},
        'uncertainty':'Paired session-then-trial bootstrap, conditional on selected checkpoints; one participant/seed; not population equivalence or selection-adjusted',
        'verdict':'artifact-relative numerical agreement' if equivalent else 'artifact-relative agreement not established'}
