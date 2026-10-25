"""Refresh E017 cached metric accounting; no raw recording access or decoding."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=row
    return previous[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
    assert not out.exists();run=ROOT/'research/runs/e017-willett-selected-final-20261009-01'
    identity=json.loads((run/'identity.json').read_text());assert digest(identity['checkpoint'])==identity['checkpoint_sha256']
    verify=json.loads((run/'verification.json').read_text())
    for name,h in verify['scored_artifact_sha256'].items():assert digest(run/name)==h
    rows=json.loads((run/'decoded-predictions.json').read_text());assert len(rows)==600
    assert len({r['trial_id']for r in rows})==600
    # Saved neural logits were independently verified at E017 completion.
    per_errors=word_errors=phone_units=word_units=0
    for row in rows:
        pe=distance(row['reference_phones'],row['predicted_phones'])
        we=distance(row['text'].split(' '),row['decoded_text'].split(' '))
        assert pe==row['per']['errors'] and we==row['wer']['errors']
        per_errors+=pe;word_errors+=we;phone_units+=len(row['reference_phones']);word_units+=len(row['text'].split(' '))
    assert (per_errors,phone_units,word_errors,word_units)==(2788,15997,678,3682)
    gate=ROOT/'reproduction/results/retraining/e017-recipe-regression-20261012-01/completion.json'
    parity=json.loads(gate.read_text());assert parity['status']=='passed'
    out.mkdir(parents=True)
    save_json(out/'reference-verification.json',dict(status='pass',trials=600,PER=per_errors/phone_units,WER=word_errors/word_units,
        counts=dict(phoneme_errors=per_errors,phones=phone_units,word_errors=word_errors,words=word_units),
        checkpoint_sha256=identity['checkpoint_sha256'],scored_files_unchanged=True,independent_scalar_edits=True,
        native_recipe_gate_sha256=digest(gate),native_recipe_gate='MPS logits/gradients/Adam update match preserved TF first-day fixture',
        scope='Cached historical offline result rechecked, not new training, inference, WER scoring or causal calibration evidence',
        raw_recordings_read=False,final_recordings_read=False,full_training_started=False,
        implementation_sha256=digest(__file__)))
    print('Reference accounting verified: PER',per_errors/phone_units,'WER',word_errors/word_units)


if __name__=='__main__':main()
