"""Exposed-test, reference-selected ensemble ceiling; never deployed accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from research.metrics import edits
from research.reporting.nbest_uncertainty import paired_oracle_interval
from verify_nbest_diagnostic import distance


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def compare(released,stopped):
    if released['true_transcripts']!=[r['text'] for r in stopped]:
        raise ValueError('Exact ordered references differ; do not infer an alignment')
    if len(released['decoded_transcripts'])!=len(stopped) or not stopped:
        raise ValueError('Trial count mismatch')
    rows=[]
    for i,(text,r) in enumerate(zip(released['decoded_transcripts'],stopped)):
        reference=r['text'].split();a=edits(reference,text.split());b=edits(reference,r['decoded_text'].split())
        assert a['errors']==distance(reference,text.split())
        assert b['errors']==distance(reference,r['decoded_text'].split())
        oracle=a if a['errors']<=b['errors'] else b
        rows.append(dict(id=r['session']+'/test/%04d'%r['trial_index'],session=r['session'],
            reference=r['text'],released_text=text,stopped_text=r['decoded_text'],
            released=a,stopped=b,top1=a,oracle=oracle,
            oracle_member='released' if a['errors']<=b['errors'] else 'stopped'))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate identities')
    units=sum(r['released']['reference_units'] for r in rows)
    counts={f:{k:sum(r[f][k] for r in rows) for k in ['errors','reference_units','substitutions','deletions','insertions']}
            for f in ['released','stopped','oracle']}
    uncertainty=paired_oracle_interval(rows)
    wins=dict(released_better=sum(r['released']['errors']<r['stopped']['errors'] for r in rows),
              stopped_better=sum(r['stopped']['errors']<r['released']['errors'] for r in rows),
              tied=sum(r['stopped']['errors']==r['released']['errors'] for r in rows),
              both_exact=sum(r['released']['errors']==r['stopped']['errors']==0 for r in rows),
              released_exact_only=sum(r['released']['errors']==0<r['stopped']['errors'] for r in rows),
              stopped_exact_only=sum(r['stopped']['errors']==0<r['released']['errors'] for r in rows))
    return dict(trials=len(rows),reference_words=units,counts=counts,
                wer={f:c['errors']/units for f,c in counts.items()},trial_complementarity=wins,
                paired_oracle_uncertainty=uncertainty,trial_accounting=rows,
                caveat='Diagnostic best-of-two uses references; no ensemble selector trained or deployed, no source ensemble reproduction')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    e000=ROOT/'reproduction/results/released-checkpoint-3gram.json'
    e008=ROOT/'research/runs/e008-stopped-wer-20261007-01/decoded-predictions.json'
    if sha(e000)!='15309213c3497524b69abf8da16633f363bed1bb2c92c9cf0faf2568df5124e9':
        raise ValueError('Released replay artifact identity differs')
    verify=json.loads((e008.parent/'verification.json').read_text())
    if sha(e008)!=verify['predictions_sha256']:raise ValueError('Stopped checkpoint decoded artifact differs')
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(scope='exposed-public-test-complementarity-diagnostic',
        inputs={str(e000.relative_to(ROOT)):sha(e000),str(e008.relative_to(ROOT)):sha(e008)},
        code_sha256=sha(Path(__file__)),trial_selection='All 600 in exact matching reference order',
        comparator='Released checkpoint first; oracle picks lower word-edit count, ties go to released',
        uncertainty='2000 paired session-then-trial draws, seed101; one participant/two fixed checkpoints, no seed dispersion',
        day_provenance='E000 has no trial IDs; inherit E008 days after exact ordered-reference matching and common released-test recipe',
        selector_trained=False,training_restarted=False,v1_restarted=False,
        scores='Acoustic/LM scores not available in saved top-1 artifacts; not fabricated'))
    released=json.loads(e000.read_text());stopped=json.loads(e008.read_text())
    result=compare(released,stopped)
    assert result['counts']['released']['errors']==695 and result['counts']['stopped']['errors']==840
    assert result['reference_words']==3682 and result['trials']==600
    save(a.output/'metrics.json',result)
    save(a.output/'verification.json',dict(status='passed-independent-scalar-edit-counts',
        trials=result['trials'],metrics_sha256=sha(a.output/'metrics.json'),protocol_sha256=sha(a.output/'protocol.json'),
        alignment='Exact reference array equality; day metadata inherited, not recovered from E000 IDs'))
    print(json.dumps({k:v for k,v in result.items() if k!='trial_accounting'},indent=2))
