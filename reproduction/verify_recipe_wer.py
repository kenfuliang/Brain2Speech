"""Independently recompute stopped-checkpoint phone/text edit accounting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from train_pytorch_recipe import ROOT, sha, write
sys.path.insert(0,str(ROOT))
from research.metrics import edits, summarize


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    identity=json.loads((a.run/'identity.json').read_text())
    metrics=json.loads((a.run/'metrics.json').read_text())
    rows=json.loads((a.run/'decoded-predictions.json').read_text())
    native=json.loads((a.run/'predictions.json').read_text())
    assert json.loads((a.run/'status.json').read_text())['status']=='complete'
    assert len(rows)==len(native)==metrics['trials']==600
    assert sha(Path(identity['checkpoint']))==identity['checkpoint_sha256']==metrics['checkpoint_sha256']
    for name,expected in identity['artifacts_sha256'].items(): assert sha(a.run/name)==expected
    assert {day:sum(r['day']==day for r in rows) for day in range(4,19)}=={day:40 for day in range(4,19)}
    for r,n in zip(rows,native):
        for key in n: assert r[key]==n[key]
        assert edits(r['text'].split(' '),r['decoded_text'].split(' '))==r['wer']
        assert edits(list(r['text']),list(r['decoded_text']))==r['cer']
        for field in ['per','wer','cer']:
            assert r[field]['errors']==sum(r[field][k] for k in ['substitutions','deletions','insertions'])
    recomputed=summarize(rows,repetitions=2000)
    for field in ['per','wer','cer']:
        assert metrics[field]==recomputed[field]
    assert np.isclose(np.mean([r['prediction_loss'] for r in rows]),metrics['mean_prediction_loss'])
    record=dict(status='passed-independent-text-accounting',trials=600,sessions=15,
                checkpoint_sha256=identity['checkpoint_sha256'],
                counts={f:{k:sum(r[f][k] for r in rows) for k in
                           ['errors','reference_units','substitutions','deletions','insertions']} for f in ['per','wer','cer']},
                metrics_sha256=sha(a.run/'metrics.json'),predictions_sha256=sha(a.run/'decoded-predictions.json'),
                caveat='Phone error totals inherited from original export; text edits and all aggregate/CI summaries independently recomputed')
    write(a.run/'verification.json',record); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
